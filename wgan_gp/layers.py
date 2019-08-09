from tensorflow.python.keras.layers.merge import _Merge
import tensorflow as tf


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        mixing_weights = tf.random.uniform(tf.shape(inputs[0]))
        return mixing_weights * inputs[0] + (1 - mixing_weights) * inputs[1]
