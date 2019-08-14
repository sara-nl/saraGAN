from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, multiply, Reshape
import tensorflow as tf


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        mixing_weights = tf.random.uniform(tf.shape(inputs[0]))
        return mixing_weights * inputs[0] + (1 - mixing_weights) * inputs[1]


class ConditionalInputEmbedding(Sequential):
    def __init__(self, n_classes, embedding_dim, **kwargs):
        super(ConditionalInputEmbedding, self).__init__(**kwargs)

        self.embedding = Embedding(n_classes, embedding_dim)

    def call(self, inputs, training=None, mask=None):

        input = inputs[0]
        label = inputs[1]

        x = self.embedding(label)
        x = Reshape(input.shape[1:])(x)
        x = multiply([x, input])

        return x


