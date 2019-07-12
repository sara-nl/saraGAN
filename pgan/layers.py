import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Conv2D, LeakyReLU, AveragePooling2D, UpSampling2D
from tensorflow.python.keras import Sequential
from initializers import Karras
from tensorflow.python.keras import backend


class PixelNorm(Layer):
    def __init__(self, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)
        self.epsilon = 1e-8

    def call(self, inputs, **kwargs):
        return inputs * tf.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
                                 + self.epsilon)


class MinibatchStandardDeviation(Layer):
    """Directly taken from NVIDIA repo."""
    def __init__(self, group_size, **kwargs):
        super(MinibatchStandardDeviation, self).__init__(**kwargs)
        self.group_size = group_size

    def call(self, inputs, **kwargs):
        with tf.variable_scope('MinibatchStdDev'):
            # Minibatch must be divisible by (or smaller than) group_size.
            x = tf.transpose(inputs, perm=(0, 3, 1, 2))
            group_size = tf.minimum(self.group_size, tf.shape(x)[0])
            s = x.shape                                             # [NCHW]  Input shape.
            y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
            y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
            y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
            y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
            y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
            y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)    # [M111]  Take average over fmaps and pixels.
            y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
            y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
            y = tf.concat([x, y], axis=1)                           # [NCHW]  Append as new fmap.
            return tf.transpose(y, perm=(0, 2, 3, 1))


class DiscriminatorBlock(Sequential):
    def __init__(self,
                 num_channels,
                 kernel_sizes,
                 strides):
        """

        Parameters
        ----------
        num_channels: tuple of channels of first and second conv2d
        kernel_sizes: tuple of kernel sizes of first and second conv2d
        strides: tuple of strides for first and second conv2d
        """
        super(DiscriminatorBlock, self).__init__()

        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        self.add(Conv2D(self.num_channels[0],
                        self.kernel_sizes[0],
                        strides=self.strides[0],
                        padding='same',
                        kernel_initializer=Karras(),
                        activation=LeakyReLU(alpha=0.2)))

        self.add(PixelNorm())

        self.add(Conv2D(self.num_channels[1],
                        self.kernel_sizes[1],
                        strides=self.strides[1],
                        padding='same',
                        kernel_initializer=Karras(),
                        activation=LeakyReLU(alpha=0.2)))

        self.add(PixelNorm())
        self.add(AveragePooling2D())


class GeneratorBlock(Sequential):
    def __init__(self,
                 num_channels,
                 kernel_sizes,
                 strides,
                 upsample=True):
        """

        Parameters
        ----------
        num_channels: tuple of channels of first and second conv2d
        kernel_sizes: tuple of kernel sizes of first and second conv2d
        strides: tuple of strides for first and second conv2d
        """
        super(GeneratorBlock, self).__init__()
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.upsample = upsample

        if self.upsample:
            self.add(UpSampling2D())

        self.add(Conv2D(self.num_channels[0],
                 self.kernel_sizes[0],
                 strides=self.strides[0],
                 padding='same',
                 kernel_initializer=Karras(),
                 activation=LeakyReLU(alpha=0.2)))

        self.add(PixelNorm())

        self.add(Conv2D(self.num_channels[1],
                 self.kernel_sizes[1],
                 strides=self.strides[1],
                 padding='same',
                 kernel_initializer=Karras(),
                 activation=LeakyReLU(alpha=0.2)))

        self.add(PixelNorm())


class InterpolationLayer(Layer):
    """Used in GradientPenaltyLoss, see Gulrajani Algorithm 1"""
    def __init__(self):
        super(InterpolationLayer, self).__init__()

    def call(self, inputs, **kwargs):
        mixing_weights = backend.random_uniform(tf.shape(inputs[0]))
        return mixing_weights * inputs[0] + (1 - mixing_weights) * inputs[1]


class AlphaMixingLayer(Layer):
    """Mixing the upscaling skip connection and forward pass during progressive training."""
    def __init__(self, alpha):
        super(AlphaMixingLayer, self).__init__()
        self.alpha = alpha

    def call(self, inputs, **kwargs):
        return self.alpha * inputs[0] + (1 - self.alpha) * inputs[1]
