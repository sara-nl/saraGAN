import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Conv2DTranspose, BatchNormalization, \
    Activation, UpSampling2D, AveragePooling2D, ReLU


class Generator(Model):
    def __init__(self, output_channels=1):
        super(Generator, self).__init__()

        self.input_block = Sequential([
            Conv2DTranspose(64, 4, strides=1, padding='valid'),
            BatchNormalization(),
            ReLU()
        ])

        self.block1 = Sequential([
            UpSampling2D(),
            Conv2D(32, 3, padding='same'),
            BatchNormalization(),
            ReLU()
        ])
        self.block2 = Sequential([
            UpSampling2D(),
            Conv2D(16, 3, padding='same'),
            BatchNormalization(),
            ReLU()
        ])

        self.torgb = Conv2DTranspose(output_channels, 4, strides=2, padding='same')
        self.tanh = Activation('tanh')

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)

    def call(self, inputs, training=None, mask=None):
        x = self.input_block(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.torgb(x)
        x = self.tanh(x)
        return x


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2D(16, 4, strides=2, padding='same')

        self.block1 = Sequential([
            AveragePooling2D(),
            Conv2D(32, 3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
        ])

        self.block2 = Sequential([
            AveragePooling2D(),
            Conv2D(64, 3, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])

        self.conv4 = Conv2D(128, 4, strides=1, padding='valid')
        self.dense1 = Dense(1)

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv4(x)
        outputs = self.dense1(x)

        return outputs
