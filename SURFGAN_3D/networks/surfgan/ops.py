# pylint: disable=unused-wildcard-import
from networks.ops import *


def modulated_conv3d(x, z, f, k, activation, up=False, demodulate=True, param=None, lrmul=1):
    """
    :param x: input
    :param z: latent
    :param f: number of feature maps
    :param k: kernel (tuple)
    """
    w = get_weight([*k, x.shape[1].value, f], activation, param=param, lrmul=lrmul)
    ww = w[np.newaxis]  # Introduce minibatch dimension.

    # Modulate.
    with tf.variable_scope('modulate'):
        s = dense(z, fmaps=x.shape[1].value, activation=activation, param=param)
        s = apply_bias(s) + 1
        s = act(s, activation, param)
        ww = ww * s[:, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

    # Demodulate.
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1, 2, 3, 4]) + 1e-8)

    x *= tf.cast(s[:, :, np.newaxis, np.newaxis, np.newaxis], x.dtype)

    if up:
        x = upscale3d(x)

    x = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')

    if demodulate:
        x *= d[:, :, np.newaxis, np.newaxis, np.newaxis]

    return x


def to_rgb(x, z, channels=1):
    x = modulated_conv3d(x, z, channels, (1, 1, 1), activation='linear', demodulate=False)
    return apply_bias(x)

