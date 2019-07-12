import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.keras import Model
from layers import PixelNorm, DiscriminatorBlock, GeneratorBlock, InterpolationLayer, MinibatchStandardDeviation, \
    AlphaMixingLayer
from initializers import Karras
from loss import WassersteinLoss, GradientPenaltyLoss, DriftLoss
from tensorflow.python.keras.layers import Dense, LeakyReLU, Reshape, Input, UpSampling2D, Conv2D, AveragePooling2D, \
    Flatten
from architectures.celeba import generator_architecture, discriminator_architecture
import numpy as np
from architectures.celeba import CELEBA_KERNEL_OFFSET
from functools import partial
from tensorflow.python.keras import backend
from tensorflow.python.keras.datasets import cifar10


class Karras2018:
    def __init__(self, phase):
        # TODO: Put as input parameters.
        self.phase = phase
        self.img_shape = 2 * 2 ** self.phase
        self.channels = 3
        self.img_shape = (self.img_shape, self.img_shape, self.channels)
        self.latent_dim = 512

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5

        hvd.init()

        # TODO: check how these should be set.
        self.d_ln_rate = 1e-4 * hvd.size()
        self.g_ln_rate = 1e-4 * hvd.size()

        g_optim, d_optim = self.initialize_optimizers(self.d_ln_rate, self.g_ln_rate)

        # TODO: Set this when doing progressive training.
        # 0: Only use interpolation. 1: Only use GAN output.
        self.alpha = 1

        # Build the generator and discriminator
        self.generator = self.build_generator(z_dim=512, architecture=generator_architecture)
        self.discriminator = self.build_discriminator(self.img_shape, discriminator_architecture)

        # DISCRIMINATOR
        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake_score = self.discriminator(fake_img)
        real_score = self.discriminator(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = InterpolationLayer()([real_img, fake_img])
        # Determine validity of weighted sample
        interpolated_score = self.discriminator(interpolated_img)

        # Loss functions
        self.gradient_penalty_loss = GradientPenaltyLoss()
        self.wasserstein_loss = WassersteinLoss()
        self.drift_loss = DriftLoss()

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        self.partial_gp_loss = partial(self.gradient_penalty_loss,
                                       averaged_samples=interpolated_img)
        self.partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                                  outputs=[real_score, fake_score, interpolated_score, real_score])

        # From paper.
        _lambda = 10
        epsilon = 1e-4
        # Weights determine direction of the gradients. See loss function algorithm 1 Gulrajani 2017
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        self.partial_gp_loss,
                                        self.drift_loss],
                                  optimizer=d_optim,
                                  loss_weights=[-1, 1, _lambda, epsilon])

        # GENERATOR
        self.generator.trainable = True
        self.discriminator.trainable = False

        # Sampled noise for input to generator
        z_gen = Input(shape=(512,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        real_score = self.discriminator(img)
        # Defines generator model
        self.generator_model = Model(z_gen, real_score)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=g_optim)

    @staticmethod
    def initialize_optimizers(d_ln_rate, g_ln_rate):

        # beta1, beta2, epsilon from the paper appendix A.2.
        with tf.variable_scope('discriminator_optimizer'):
            d_optim = tf.train.AdamOptimizer(
                learning_rate=d_ln_rate,
                beta1=0.0,
                beta2=0.99,
                epsilon=1e-8,
                name='d_optim')
            d_optim = hvd.DistributedOptimizer(d_optim)

        with tf.variable_scope('generator_optimizer'):
            g_optim = tf.train.AdamOptimizer(
                learning_rate=g_ln_rate,
                beta1=0.0,
                beta2=0.99,
                epsilon=1e-8,
                name='g_optim')
            g_optim = hvd.DistributedOptimizer(g_optim)

        return d_optim, g_optim

    def build_generator(self, z_dim, architecture):

        z = Input((z_dim,))
        x = PixelNorm()(z)
        base_features = architecture[1]['channels'][0]

        x = Dense(4 * 4 * base_features,
                  activation=LeakyReLU(alpha=0.2),
                  kernel_initializer=Karras(gain=np.sqrt(2) / 4))(x)

        x = Reshape((4, 4, base_features))(x)
        x = PixelNorm()(x)

        x_upscale = None

        for i in range(1, self.phase + 1):

            kernel_sizes = architecture[i]['filters']

            if i == 1:
                upsample = False
                # TODO: add flag.
                kernel_sizes[0] += CELEBA_KERNEL_OFFSET

            else:
                upsample = True

            x = GeneratorBlock(num_channels=architecture[i]['channels'],
                               kernel_sizes=kernel_sizes,
                               strides=((1, 1), (1, 1)),
                               upsample=upsample)(x)
            if i == self.phase - 1:
                x_upscale = UpSampling2D()(x)

        x = Conv2D(3, 1)(x)

        if x_upscale is not None:
            img_upscaled = Conv2D(3, 1)(x_upscale)
            x = AlphaMixingLayer(self.alpha)((x, img_upscaled))

        model = Model(inputs=z, outputs=x)

        model.summary()

        return model

    def build_discriminator(self, img_shape, architecture):

        image = Input(shape=img_shape)

        # the 'FromRGB' operation.
        x = Conv2D(filters=architecture[len(architecture) - 1]['channels'][1],
                   kernel_size=1,
                   activation=LeakyReLU(alpha=0.2),
                   kernel_initializer=Karras())(image)

        img_downsized = AveragePooling2D()(x)

        print("Discriminator")
        # Construct graph.
        for i in range(len(architecture) - self.phase + 1, len(architecture)):

            x = DiscriminatorBlock(num_channels=architecture[i]['channels'],
                                   kernel_sizes=architecture[i]['filters'],
                                   strides=((1, 1), (1, 1)))(x)

            # FromRGB
            if i == len(architecture) - self.phase + 1:
                x_downsized = Conv2D(filters=architecture[i]['channels'][1],
                                     kernel_size=1,
                                     activation=LeakyReLU(alpha=0.2),
                                     kernel_initializer=Karras())(img_downsized)
                print("Mixing")
                x = AlphaMixingLayer(self.alpha)((x, x_downsized))

        x = MinibatchStandardDeviation(group_size=4)(x)

        x = Conv2D(filters=architecture[len(architecture)]['channels'][0],
                   kernel_size=architecture[len(architecture)]['filters'][0],
                   activation=LeakyReLU(alpha=0.2),
                   kernel_initializer=Karras(),
                   padding='same')(x)

        # Last Conv2D layer (no padding) brings output shape to [1, 1, 512]
        x = Conv2D(filters=architecture[len(architecture)]['channels'][1],
                   kernel_size=architecture[len(architecture)]['filters'][1] + CELEBA_KERNEL_OFFSET,
                   activation=LeakyReLU(alpha=0.2),
                   kernel_initializer=Karras())(x)

        # Gain = 1 taken from face embedding GAN.
        x = Flatten()(x)
        x = Dense(units=1,
                  kernel_initializer=Karras(gain=1))(x)

        model = Model(inputs=image, outputs=x)

        model.summary()

        return model

    def train(self, epochs, batch_size):

        # Load the dataset
        (x_train, _), (_, _) = cifar10.load_data()

        # Rescale -1 to 1
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        # Adversarial ground truths
        dummy = np.zeros((batch_size, 1))  # Dummy for y_true in Keras loss functions.

        assert self.n_critic > 0

        for epoch in range(epochs):
            session = backend.get_session()
            session.run(hvd.broadcast_global_variables(0))

            # Placeholder
            d_loss = None

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                imgs = x_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                          [dummy, dummy, dummy, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.generator_model.train_on_batch(noise, dummy)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))


if __name__ == '__main__':
    # Phase 4: train directly on 32x32 (CIFAR10)
    wgan = Karras2018(phase=4)
    wgan.train(epochs=30000, batch_size=32)
