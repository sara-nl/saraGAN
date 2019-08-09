import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Conv2DTranspose, BatchNormalization, \
    Activation, UpSampling2D, AveragePooling2D, ReLU, Input, LayerNormalization


def build_generator(latent_dim,
                    output_channels=3,
                    base_dim=64,
                    n_upsamplings=3,
                    norm='batch_norm'):

    norm = BatchNormalization
    inputs = Input(shape=(1, 1, latent_dim))

    d = min(base_dim * 2 ** (n_upsamplings - 1), base_dim * 8)
    x = Conv2DTranspose(d, 4, strides=1, padding='valid', use_bias=False)(inputs)
    x = norm()(x)
    x = ReLU()(x)

    for i in range(n_upsamplings - 1):
        d = min(base_dim * 2 ** (n_upsamplings - 2 - i), base_dim * 8)
        x = Conv2DTranspose(d, 4, strides=2, padding='same', use_bias=False)(x)
        x = norm()(x)
        x = ReLU()(x)

    x = Conv2DTranspose(output_channels, 4, strides=2, padding='same')(x)
    x = Activation('tanh')(x)

    return Model(inputs=inputs, outputs=x, name='Generator')


def build_discriminator(input_shape=(32, 32, 3),
                        latent_dim=128,
                        base_dim=64,
                        n_downsamplings=3,
                        norm='batch_norm'):
    norm = LayerNormalization

    inputs = Input(shape=input_shape)

    x = Conv2D(base_dim, 4, strides=2, padding='same')(inputs)
    x = ReLU()(x)

    for i in range(n_downsamplings - 1):
        d = min(base_dim * 2 ** (i + 1), base_dim * 8)
        x = Conv2D(d, 4, strides=2, padding='same')(x)
        x = norm()(x)
        x = ReLU()(x)

    x = Conv2D(1, 4, strides=1, padding='valid')(x)

    return Model(inputs=inputs, outputs=x, name='Discriminator')
