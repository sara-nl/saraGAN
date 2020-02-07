# pylint: disable=unused-wildcard-import
from networks.ops import *


# def get_weight(shape, activation, lrmul=1, param=None) -> tuple:
#     fan_in = np.prod(shape[:-1])
#     gain = calculate_gain(activation, param)
#     he_std = gain / np.sqrt(fan_in)
#     init_std = 1.0 / lrmul
#     runtime_coef = he_std * lrmul
#     return tf.get_variable('weight', shape=shape,
#                            initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef, runtime_coef

def get_weight(shape, activation, lrmul=1, use_eq_lr=True, use_spectral_norm=False, param=None):
    fan_in = np.prod(shape[:-1])
    gain = calculate_gain(activation, param)
    he_std = gain / np.sqrt(fan_in)
    runtime_coef = he_std * lrmul if use_eq_lr else lrmul
    init_std = 1.0 / lrmul
    w = tf.get_variable('weight', shape=shape,
                        initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef

    if use_spectral_norm:
        w = spectral_norm(w)

    return w, runtime_coef

def apply_noise(x, runtime_coef):
    assert len(x.shape) == 4  # NCHW
    with tf.variable_scope('apply_noise'):
        noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]])
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros()) * runtime_coef
        return x + noise * noise_strength


def apply_bias(x, runtime_coef):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.random_normal()) * runtime_coef
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])


def dense(x, fmaps, activation, lrmul=1, param=None):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w, runtime_coef = get_weight([x.shape[1].value, fmaps], activation, lrmul=lrmul, param=param)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w), runtime_coef


def conv2d(x, fmaps, kernel, activation, param=None, lrmul=1):
    w, runtime_coef = get_weight([*kernel, x.shape[1].value, fmaps], activation, param=param, lrmul=lrmul)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW'), runtime_coef


def modulated_conv2d(x, z, f, k, activation, up=False, demodulate=True, param=None, lrmul=1):
    """
    :param x: input
    :param z: latent
    :param f: number of feature maps
    :param k: kernel (tuple)
    """
    w, runtime_coef = get_weight([*k, x.shape[1].value, f], activation, param=param, lrmul=lrmul)
    ww = w[np.newaxis]  # Introduce minibatch dimension.

    # Modulate.
    with tf.variable_scope('modulate'):
        s, runtime_coef_dense = dense(z, fmaps=x.shape[1].value, activation=activation, param=param)
        s = apply_bias(s, runtime_coef_dense) + 1
        s = act(s, activation, param)
        ww = ww * s[:, np.newaxis, np.newaxis, :, np.newaxis]

    # Demodulate.
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1, 2, 3]) + 1e-8)

    x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype)

    if up:
        x = upscale2d(x)

    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')

    if demodulate:
        x *= d[:, :, np.newaxis, np.newaxis]

    return x, runtime_coef


def from_rgb(x, filters_out, activation, param=None):
    x, runtime_coef = conv2d(x, filters_out, (1, 1), activation, param)
    x = apply_bias(x, runtime_coef)
    x = act(x, activation, param=param)
    return x


def to_rgb(x, z, channels=3):
    x, runtime_coef = modulated_conv2d(x, z, channels, (1, 1), activation='linear', demodulate=False)
    return apply_bias(x, runtime_coef)

