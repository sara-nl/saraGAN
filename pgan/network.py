import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras import layers
import math
import horovod.tensorflow as hvd

from loss import wasserstein_loss, gradient_penalty_loss

def num_filters(phase, num_phases, base_dim):
    num_downscales = int(math.log2(base_dim / 16))
    filters = min(base_dim // (2 ** (phase - num_phases + num_downscales)), base_dim)
    return filters


class GeneratorBlock(keras.Sequential):
    def __init__(self, filters, **kwargs):
        super(GeneratorBlock, self).__init__(**kwargs)
        self.add(layers.UpSampling3D())
        self.add(layers.Convolution3D(filters, (3, 3, 3), padding='same'))
        self.add(layers.LeakyReLU())

        self.add(layers.Convolution3D(filters, (3, 3, 3), padding='same'))
        self.add(layers.LeakyReLU())
        
        # self.add(layers.Convolution3D(filters, (3, 3, 3), padding='same'))
        # self.add(layers.LeakyReLU())
        

def make_generator(phase, num_phases, base_dim, latent_dim):
        
    z = layers.Input((latent_dim,), name='generator_input')
    alpha_in = layers.Input((1,), name='generator_mixing_parameter')
    
    filters = num_filters(0, num_phases, base_dim)
    
    x = keras.Sequential((
        layers.Dense(4 * 4 * 4 * filters),
        layers.LeakyReLU(),
        layers.Reshape((4, 4, 4, filters)),
        layers.Convolution3D(filters, (3, 3, 3), padding='same'),
        layers.LeakyReLU(),
        # layers.Convolution3D(filters, (3, 3, 3), padding='same'),
        # layers.LeakyReLU()
    ), name='generator_in')(z)
    
    x_upsampled = None  # Placeholder
    for i in range(1, phase):
        
        if i == phase - 1:
            x_upsampled = layers.LeakyReLU()(layers.Conv3D(
                1, 1, name=f'to_rgb_{phase - 1}')(layers.UpSampling3D()(x)))
        
        filters = num_filters(i, num_phases, base_dim)
        x = GeneratorBlock(filters=filters, name=f'generator_block_{i}')(x)
            
    x = layers.Conv3D(1, 1, name=f'to_rgb_{phase}')(x)
    
    if x_upsampled is not None:
        x = AlphaMixingLayer()([x_upsampled, x, alpha_in])
    
    model = keras.Model(inputs=((z, alpha_in)), outputs=(x))
    return model
        
        
class DiscriminatorBlock(keras.Sequential):
    def __init__(self, filters, **kwargs):
        super(DiscriminatorBlock, self).__init__(**kwargs)
        self.add(layers.Conv3D(filters, 3, padding='same'))
        self.add(layers.LeakyReLU())
        
        # self.add(layers.Conv3D(filters, 3, padding='same'))
        # self.add(layers.LeakyReLU())

        self.add(layers.Conv3D(filters, 3, padding='same'))
        self.add(layers.LeakyReLU())

        self.add(layers.AveragePooling3D())


def make_discriminator(phase, num_phases, base_dim, img_shape, latent_dim):
    
    z = layers.Input(img_shape, name='discriminator_input')
    alpha = layers.Input((1,), name='discriminator_mixing_parameter')
        
    filters = num_filters(phase - 1, num_phases, base_dim)
    x = layers.Conv3D(filters, 1, name=f'from_rgb_{phase}')(z)
    x = layers.LeakyReLU()(x)
    

    for i in reversed(range(0, phase - 1)):
        filters = num_filters(i, num_phases, base_dim)
        
        x = DiscriminatorBlock(filters=filters,
                               name=f'discriminator_block_{i + 1}')(x)
        
        if i == phase - 2:
            x_downscaled = layers.LeakyReLU()(layers.Conv3D(filters, 1, name=f'from_rgb_{phase - 1}')(layers.AveragePooling3D()(z)))
            x = AlphaMixingLayer()([x_downscaled, x, alpha])
    
    x = keras.Sequential((
        # layers.Conv3D(filters, (3, 3, 3), padding='same'), 
        # layers.LeakyReLU(),
        layers.Conv3D(filters, (3, 3, 3), padding='same'), 
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dense(latent_dim),
        layers.LeakyReLU(),
        layers.Dense(1),
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

        if horovod and is_first_batch:
            if hvd.rank() == 0:
                print("Broacasting discriminator variables.")
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



    
