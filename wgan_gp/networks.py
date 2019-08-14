import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Conv2DTranspose, BatchNormalization, \
    Activation, UpSampling2D, AveragePooling2D, ReLU, Input, LayerNormalization, Embedding, multiply, Reshape, Flatten, Concatenate, LeakyReLU
import numpy as np

from layers import ConditionalInputEmbedding


def build_generator(latent_dim,
                    img_shape,
                    base_dim,
                    n_classes=None):

    norm = BatchNormalization
    latent_space = Input(shape=(latent_dim,), name='latent_space')

    if n_classes:

        label = Input(shape=(n_classes,), name='label_generator')
        x = Concatenate()([latent_space, label])
    else:
        x = latent_space

    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=1)

    n_upsamplings = int(np.log2(img_shape[1])) - 2
    d = min(base_dim * 2 ** (n_upsamplings - 1), base_dim * 8)
    x = Conv2DTranspose(d, 4, strides=2, padding='valid', use_bias=False)(x)
    x = norm()(x)
    x = ReLU()(x)
    for i in range(n_upsamplings - 1):
        d = min(base_dim * 2 ** (n_upsamplings - 2 - i), base_dim * 8)
        x = Conv2DTranspose(d, 4, strides=2, padding='same', use_bias=False)(x)
        x = norm()(x)
        x = ReLU()(x)

    output_channels = img_shape[-1]
    x = Conv2DTranspose(output_channels, 4, strides=2, padding='same')(x)
    x = Activation('tanh')(x)

    if n_classes:
        model = Model(inputs=[latent_space, label], outputs=x, name='Generator')

    else:
        model = Model(inputs=latent_space, outputs=x, name='Generator')

    return model


def build_discriminator(img_shape,
                        latent_dim,
                        base_dim,
                        n_classes):
    norm = LayerNormalization

    inputs = Input(shape=img_shape, name='image')

    x = Conv2D(base_dim, 4, strides=2, padding='same')(inputs)
    x = LeakyReLU()(x)

    if n_classes:
        label = Input(shape=(n_classes,), name='label_discriminator')
        discriminator_condition_embedded = Reshape((1, 1, n_classes))(label)
        discriminator_condition_embedded = Conv2DTranspose(1,
                                                           kernel_size=(x.shape[1], x.shape[2]),
                                                           strides=2,
                                                           padding='valid')(discriminator_condition_embedded)
        discriminator_condition_embedded = LeakyReLU()(discriminator_condition_embedded)

        x = Concatenate()([discriminator_condition_embedded, x])

    n_downsamplings = int(np.log2(img_shape[1])) - 2
    for i in range(n_downsamplings - 1):
        d = min(base_dim * 2 ** (i + 1), base_dim * 8)
        x = Conv2D(d, 4, strides=2, padding='same')(x)
        x = norm()(x)
        x = LeakyReLU()(x)

    x = Conv2D(latent_dim, 4, strides=1, padding='valid')(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(1)(x)

    if n_classes is not None:
        model = Model(inputs=[inputs, label], outputs=x, name='Discriminator')
    else:
        model = Model(inputs=inputs, outputs=x, name='Discriminator')

    return model
