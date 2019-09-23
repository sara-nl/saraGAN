import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras import layers
import numpy as np
import horovod.tensorflow as hvd

from loss import wasserstein_loss, gradient_penalty_loss

def num_filters(phase, num_phases, base_dim):
    num_downscales = int(np.log2(base_dim / 16))
    filters = min(base_dim // (2 ** (phase - num_phases + num_downscales)), base_dim)
    return filters

class ChannelNormalization(tf.keras.layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(ChannelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + 1e-8)

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape


def get_wscale(shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1])
    wscale = gain / (fan_in ** (1 / 2)) # He init
    # wscale = gain / np.sqrt(fan_in) # He init
    return wscale


class Conv3D(tf.keras.layers.Layer):
    def __init__(self, channels_out, kernel, 
                 strides=(1, 1, 1, 1, 1), padding='SAME', gain=np.sqrt(2), **kwargs):
        super(Conv3D, self).__init__(**kwargs)
        assert isinstance(kernel, int)
        self.kernel = kernel
        self.channels_out = channels_out
        self.gain = gain
        self.strides = strides
        self.padding = padding.upper()
    
    
    def build(self, input_shape):
        weight_shape = [self.kernel, self.kernel, self.kernel, input_shape[-1], self.channels_out]
        self.wscale = get_wscale(weight_shape, self.gain)
        initializer = tf.random_normal_initializer(stddev=self.wscale)
        self.kernel = self.add_weight("kernel",
                                      initializer=initializer,
                                      shape=weight_shape)
        initializer = tf.zeros_initializer()
        self.bias = self.add_weight("bias",
                                    initializer=initializer,
                                    shape=(self.channels_out,))
        
    def call(self, inputs):
        return tf.nn.conv3d(inputs, self.kernel * self.wscale, self.strides, self.padding) + self.bias

    
class Dense(tf.keras.layers.Layer):
    def __init__(self, channels_out, gain=np.sqrt(2), **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.channels_out = channels_out
        self.gain = gain

    def build(self, input_shape):
        weight_shape = (input_shape[-1], self.channels_out)
        self.wscale = get_wscale(weight_shape, self.gain)
        initializer = tf.random_normal_initializer()
        self.kernel = self.add_weight("kernel",
                                      initializer=initializer,
                                      shape=weight_shape)
        
        initializer = tf.zeros_initializer()
        self.bias = self.add_weight("bias",
                                    initializer=initializer,
                                    shape=(self.channels_out,))

    def call(self, input):
        return tf.matmul(input, self.kernel * self.wscale) + self.bias



def conv3d(filters, kernel, padding='valid', **kwargs):
    return Conv3D(filters, kernel, padding=padding, **kwargs)


def dense(channels_out, **kwargs):
    # return layers.Dense(channels_out, **kwargs)
    return Dense(channels_out, **kwargs)


class GeneratorBlock(keras.Sequential):
    def __init__(self, filters, **kwargs):
        super(GeneratorBlock, self).__init__(**kwargs)
        self.add(layers.UpSampling3D())
        self.add(conv3d(filters, 3, padding='same'))
        self.add(layers.LeakyReLU())
        self.add(ChannelNormalization())

        self.add(conv3d(filters, 3, padding='same'))
        self.add(layers.LeakyReLU())
        self.add(ChannelNormalization())


def make_generator(phase, num_phases, base_dim, latent_dim):
        
    z = layers.Input((latent_dim,), name='generator_input')
    alpha_in = layers.Input((1,), name='generator_mixing_parameter')
    
    filters = num_filters(0, num_phases, base_dim)
    
    x = keras.Sequential((
        dense(4 * 4 * 4 * filters, gain=np.sqrt(2) / 4),
        layers.LeakyReLU(),
        layers.Reshape((4, 4, 4, filters)),
        conv3d(filters, 3, padding='same'),
        layers.LeakyReLU(),
        ChannelNormalization(),
    ), name='generator_in')(z)
    
    x_upsampled = None  # Placeholder
    for i in range(1, phase):
        
        if i == phase - 1:
            x_upsampled = layers.LeakyReLU()(conv3d(
                1, 1, gain=1, name=f'to_rgb_{phase - 1}')(layers.UpSampling3D()(x)))
        
        filters = num_filters(i, num_phases, base_dim)
        x = GeneratorBlock(filters=filters, name=f'generator_block_{i}')(x)
            
    x = conv3d(1, 1, gain=1, name=f'to_rgb_{phase}')(x)
    
    if x_upsampled is not None:
        x = AlphaMixingLayer()([x_upsampled, x, alpha_in])
    
    model = keras.Model(inputs=((z, alpha_in)), outputs=(x))
    return model
        

class MinibatchStandardDeviation(tf.keras.layers.Layer):
    def __init__(self, group_size=4):
        super(MinibatchStandardDeviation, self).__init__()
        self.group_size = group_size
        
    def call(self, inputs):
        group_size = tf.minimum(self.group_size, tf.shape(inputs)[0])       # Minibatch must be divisible by (or smaller than) group_size.
        s = inputs.shape                                                    # [NDHWC]  Input shape.
        y = tf.reshape(inputs, [group_size, -1, s[1], s[2], s[3], s[4]])    # [GMDHWC] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                                          # [GMDHWC] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)                       # [GMDHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                            # [MDHWC]  Calc variance over group.)
        y = tf.sqrt(y + 1e-8)                                               # [MDHWC]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3, 4], keepdims=True)               # [M1111]  Take average over fmaps and pixels.
        y = tf.cast(y, inputs.dtype)                                        # [M1111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], s[3], 1])                   # [N1DHW]  Replicate over group and pixels.
        return tf.concat([inputs, y], axis=-1)                              # [NDHWC]  Append as new fmap.

        
class DiscriminatorBlock(keras.Sequential):
    def __init__(self, filters, **kwargs):
        super(DiscriminatorBlock, self).__init__(**kwargs)
        self.add(conv3d(filters, 3, padding='same'))
        self.add(layers.LeakyReLU())
        
        self.add(conv3d(filters, 3, padding='same'))
        self.add(layers.LeakyReLU())

        self.add(layers.AveragePooling3D())


def make_discriminator(phase, num_phases, base_dim, img_shape, latent_dim):
    
    z = layers.Input(img_shape, name='discriminator_input')
    alpha = layers.Input((1,), name='discriminator_mixing_parameter')
        
    filters = num_filters(phase - 1, num_phases, base_dim)
    x = conv3d(filters, 1, name=f'from_rgb_{phase}')(z)
    x = layers.LeakyReLU()(x)
    

    for i in reversed(range(0, phase - 1)):
        filters = num_filters(i, num_phases, base_dim)
        
        x = DiscriminatorBlock(filters=filters,
                               name=f'discriminator_block_{i + 1}')(x)
        
        if i == phase - 2:
            x_downscaled = layers.LeakyReLU()(conv3d(filters, 1, name=f'from_rgb_{phase - 1}')(layers.AveragePooling3D()(z)))
            x = AlphaMixingLayer()([x_downscaled, x, alpha])
    
    x = keras.Sequential((
        MinibatchStandardDeviation(),
        conv3d(filters, 3, padding='same'), 
        layers.LeakyReLU(),
        layers.Flatten(),
        dense(latent_dim),
        layers.LeakyReLU(),
        dense(1, gain=1),
    ), name='discriminator_out')(x)

    
    model = keras.Model(inputs=(z, alpha), outputs=(x,))
    return model


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = tf.random.normal((inputs[0].shape[0], 1, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    
    
class AlphaMixingLayer(_Merge):
    def _merge_function(self, inputs):
        alpha = inputs[-1]
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def get_training_functions(horovod):
    @tf.function
    def train_discriminator(generator, 
                            discriminator, 
                            discriminator_optim, 
                            batch_size, 
                            latent_dim, 
                            x_real, 
                            alpha, 
                            gradient_penalty_weight,
                            is_first_batch,
                            horovod=horovod):
        
        for layer in discriminator.layers:
            layer.trainable = True
        discriminator.trainable = True

        for layer in generator.layers:
            layer.trainable = False
        generator.trainable = False

        with tf.GradientTape() as tape:
            z = tf.random.normal(shape=(batch_size, latent_dim))
            d_real = discriminator([x_real, alpha])
            x_fake = generator([z, alpha])
            d_fake = discriminator([x_fake, alpha])

            averaged_samples = RandomWeightedAverage()([x_real, x_fake])

            d_avg = discriminator([averaged_samples, alpha])

            real_loss = wasserstein_loss(d_real)
            fake_loss = wasserstein_loss(d_fake)        
            gp_loss = gradient_penalty_loss(d_avg,
                                 averaged_samples,
                                 gradient_penalty_weight)

            d_loss = real_loss - fake_loss + gp_loss
            
        if horovod:
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optim.apply_gradients(zip(grads, discriminator.trainable_variables))
        
        # WHY DOES THIS GET PRINTED TWICE?
        if horovod and is_first_batch:
            if hvd.rank() == 0:
                print("Broadcasting discriminator variables.")
            hvd.broadcast_variables(discriminator.variables, root_rank=0)
            hvd.broadcast_variables(discriminator_optim.variables(), root_rank=0)
        
        return d_loss
    
    @tf.function
    def train_generator(generator, 
                        discriminator, 
                        generator_optim, 
                        batch_size, 
                        latent_dim, 
                        alpha, 
                        is_first_batch,
                        horovod=horovod):

        for layer in discriminator.layers:
            layer.trainable = False
        discriminator.trainable = False

        for layer in generator.layers:
            layer.trainable = True
        generator.trainable = True

        with tf.GradientTape() as tape:
            z = tf.random.normal(shape=(batch_size, latent_dim))
            x_fake = generator([z, alpha])
            d_fake = discriminator([x_fake, alpha])

            g_loss = wasserstein_loss(d_fake)

            
        if horovod:
            tape = hvd.DistributedGradientTape(tape)
        
        grads = tape.gradient(g_loss, generator.trainable_variables)
        generator_optim.apply_gradients(zip(grads, generator.trainable_variables))
        
        if horovod and is_first_batch:
            if hvd.rank() == 0:
                print("Broadcasting generator variables.")
            hvd.broadcast_variables(generator.variables, root_rank=0)
            hvd.broadcast_variables(generator_optim.variables(), root_rank=0)

        return g_loss, x_fake
    
    return train_generator, train_discriminator



    
