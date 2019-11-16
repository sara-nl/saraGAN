import tensorflow as tf
import numpy as np


def num_filters(phase, num_phases, base_dim, min_filters=32):
    num_downscales = int(np.log2(base_dim / min_filters))
    filters = min(base_dim // (2 ** (phase - num_phases + num_downscales)), base_dim)
    print(filters)
    return filters


def calculate_gain(activation, param=None):
    if activation == 'leaky_relu':
        assert param is not None

    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if activation in linear_fns or activation == 'sigmoid':
        return 1
    elif activation == 'tanh':
        return 5.0 / 3
    elif activation == 'relu':
        return np.sqrt(2.0)
    elif activation == 'leaky_relu':
        if not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return np.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(activation))


def get_weight(shape, activation, lrmul=1, param=None):
    fan_in = np.prod(shape[:-1])
    gain = calculate_gain(activation, param)
    he_std = gain / np.sqrt(fan_in)

    init_std = 1.0 / lrmul
    runtime_coef = he_std * init_std
    return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef


def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1, 1])


def dense(x, fmaps, activation, lrmul=1, param=None):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], activation, lrmul=lrmul, param=param)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)


def conv3d(x, fmaps, kernel, activation, param=None):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], activation, param=param)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')


def act(x, activation, param=None):
    if activation == 'leaky_relu':
        assert param is not None
        x = tf.nn.leaky_relu(x, alpha=param)

    return x


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('pixel_norm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 5  # NCDHW
    with tf.variable_scope('instance_norm'):
        x -= tf.reduce_mean(x, axis=[2, 3, 4], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2, 3, 4], keepdims=True) + epsilon)
        return x


def apply_noise(x, noise_var=None, randomize_noise=True):
    assert len(x.shape) == 5  # NCDHW
    with tf.variable_scope('apply_noise'):
        if noise_var is None or randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3], x.shape[4]])
        else:
            noise = noise_var
        weight = tf.get_variable('weight', shape=[x.shape[1].value], initializer=tf.initializers.zeros())
        return x + noise * tf.reshape(weight, [1, -1, 1, 1, 1])


def style_mod(x, dlatent, activation, param=None):
    with tf.variable_scope('style_mod'):
        style = dense(dlatent, fmaps=x.shape[1] * 2, activation=activation, param=param)
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        return x * (style[:, 0] + 1) + style[:, 1]


def avg_unpool3d(x, factor=2, gain=1):

    if gain != 1:
        x = x * gain

    if factor == 1:
        return x

    x = tf.transpose(x, [2, 3, 4, 1, 0])  # [B, C, D, H, W] -> [D, H, W, C, B]
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [factor ** 3, 1, 1, 1, 1, 1])
    x = tf.batch_to_space_nd(x, [factor, factor, factor], [[0, 0], [0, 0], [0, 0]])
    x = tf.transpose(x[0], [4, 3, 0, 1, 2])  # [D, H, W, C, B] -> [B, C, D, H, W]
    return x


def avg_pool3d(x, factor=2, gain=1):
    if gain != 1:
        x *= gain

    if factor == 1:
        return x

    ksize = [1, 1, factor, factor, factor]
    return tf.nn.avg_pool3d(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCDHW')


def upscale3d(x, factor=2):
    with tf.variable_scope('upscale_3d'):
        @tf.custom_gradient
        def func(x):
            y = avg_unpool3d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = avg_pool3d(dy, factor, gain=factor ** 3)
                return dx, lambda ddx: avg_pool3d(ddx, factor)
            return y, grad
        return func(x)


def downscale3d(x, factor=2):
    with tf.variable_scope('downscale_3d'):
        @tf.custom_gradient
        def func(x):
            y = avg_pool3d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = avg_unpool3d(dy, factor, gain=1 / factor ** 3)
                return dx, lambda ddx: avg_pool3d(ddx, factor)
            return y, grad
        return func(x)


def to_rgb(x, channels=1):
    return apply_bias(conv3d(x, channels, 1, activation='linear'))


def from_rgb(x, filters_out, activation, param=None):
    x = conv3d(x, filters_out, 1, activation, param)
    x = apply_bias(x)
    x = act(x, activation, param)
    return x


def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('minibatch_std'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])
        s = x.shape
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3], s[4]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[1, 2, 3, 4], keepdims=True)
        y = tf.cast(y, x.dtype)
        y = tf.tile(y, [group_size, 1, s[2], s[3], s[4]])
        return tf.concat([x, y], axis=1)


# def layer_epilogue(x, dlatents_in, noise_inputs, layer_idx, activation, param):
#     x = apply_noise(x, noise_inputs[layer_idx - 2], randomize_noise=True)
#     x = apply_bias(x)
#     x = act(x, activation, param)
#     x = instance_norm(x)
#     x = style_mod(x, dlatents_in[:, layer_idx - 2])
#     return x


if __name__ == '__main__':
    num_phases = 8
    for phase in range(1, num_phases + 1):
        nf = num_filters(phase, num_phases, base_dim=256)

        print(f'res: {2 * 2 ** phase}, nf: {nf}')

