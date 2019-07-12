"""Initializers as described in the paper."""
import tensorflow as tf
from tensorflow.python.keras.initializers import Initializer
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.init_ops import _compute_fans
import numpy as np


class Karras(Initializer):
    def __init__(self,
                 gain=np.sqrt(2),
                 use_wscale=True,
                 seed=None,
                 dtype=tf.dtypes.float32):
        """Initializer as specified in the paper.

        Parameters
        ----------
        seed: int
        gain: He initializer gain factor (np.sqrt(2))
        use_wscale: Use weight scaling as specified in section 4.1.
        """
        self.seed = seed
        self.gain = gain
        self.use_wscale = use_wscale
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        fan_in, fan_out = _compute_fans(shape)
        std = self.gain / np.sqrt(fan_in)

        if self.use_wscale:
            wscale = tf.constant(np.float32(std), name='wscale')
            return wscale * random_ops.random_normal(shape, 0, 1, dtype, seed=self.seed)

        else:
            # He init.
            return random_ops.random_normal(shape, 0, std, dtype, seed=self.seed)
