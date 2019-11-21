import tensorflow as tf
import numpy as np
import math


def calculate_gain(activation, param=None):
    if activation == 'leaky_relu':
        assert param is not None

    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if activation in linear_fns or activation == 'sigmoid':
        return 1
    elif activation == 'tanh':
        return 5.0 / 3
    elif activation == 'relu':
        return math.sqrt(2.0)
    elif activation == 'leaky_relu':
        if not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(activation))


def get_weight(shape, activation, param=None):
    fan_in = np.prod(shape[:-1])
    gain = calculate_gain(activation, param)
    std = gain / np.sqrt(fan_in)
    return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * std


def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1, 1])


def dense(x, fmaps, activation, param=None):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], activation, param=param)
    w = tf.cast(w, x.dtype)
    return apply_bias(tf.matmul(x, w))


def conv3d(x, fmaps, kernel, activation, param=None):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], activation, param)
    w = tf.cast(w, x.dtype)
    return apply_bias(tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW'))


def num_filters(phase, num_phases, base_dim):
    num_downscales = int(np.log2(base_dim / 32))
    filters = min(base_dim // (2 ** (phase - num_phases + num_downscales)), base_dim)
    print(filters)
    return filters


def generator_in(x, filters, shape, activation, param=None):
    with tf.variable_scope('dense'):
        x = dense(x, np.product(shape) * filters, activation, param)
        x = tf.nn.leaky_relu(x, alpha=param)
    x = tf.reshape(x, [-1, filters] + list(shape))
    with tf.variable_scope('conv'):
        x = conv3d(x, filters, 3, activation, param)
        x = tf.nn.leaky_relu(x, alpha=param)
        x = pixel_norm(x)
    return x


def to_rgb(x, channels=1):
    return conv3d(x, channels, 1, activation='linear')


# def upsample(x, factor=2):
#     s = x.shape
#     x = tf.reshape(x, [tf.shape(x)[0], s[1], s[2], 1, s[3], 1, s[4], 1])
#     x = tf.tile(x, [1, 1, 1, factor, 1, factor, 1, factor])
#     x = tf.reshape(x, [tf.shape(x)[0], s[1], s[2] * factor, s[3] * factor, s[4] * factor])
#     return x


def avg_unpool3d(x, factor=2):
    #   x = tf.transpose(x, [1, 2, 3, 0]) # [B, H, W, C] -> [H, W, C, B]
    x = tf.transpose(x, [2, 3, 4, 1, 0])  # [B, C, D, H, W] -> [D, H, W, C, B]
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [factor ** 3, 1, 1, 1, 1, 1])
    x = tf.batch_to_space_nd(x, [factor, factor, factor], [[0, 0], [0, 0], [0, 0]])
    #   x = tf.transpose(x[0], [3, 0, 1, 2]) # [H, W, C, B] -> [B, H, W, C]
    x = tf.transpose(x[0], [4, 3, 0, 1, 2])  # [D, H, W, C, B] -> [B, C, D, H, W]

    return x


upsample = avg_unpool3d


def generator_block(x, filters_out, activation, param=None):
    with tf.variable_scope('upsample'):
        x = upsample(x)

    with tf.variable_scope('conv_1'):
        x = conv3d(x, filters_out, 3, activation, param)
        x = tf.nn.leaky_relu(x, alpha=param)
        x = pixel_norm(x)

    with tf.variable_scope('conv_2'):
        x = conv3d(x, filters_out, 3, activation, param)
        x = tf.nn.leaky_relu(x, alpha=param)
        x = pixel_norm(x)
    return x


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('pixel_norm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


def generator(x, alpha, phase, num_phases, base_dim, base_shape, activation, param=None):
    with tf.variable_scope('generator'):
        with tf.variable_scope('generator_in'):
            x = generator_in(x, filters=base_dim, shape=base_shape[1:], activation=activation, param=param)

        x_upsample = None

        for i in range(2, phase + 1):

            if i == phase:
                with tf.variable_scope(f'to_rgb_{phase - 1}'):
                    x_upsample = upsample(to_rgb(x, channels=base_shape[0]))

            filters_out = num_filters(i, num_phases, base_dim)
            with tf.variable_scope(f'generator_block_{i}'):
                x = generator_block(x, filters_out, activation=activation, param=param)

        with tf.variable_scope(f'to_rgb_{phase}'):
            x_out = to_rgb(x, channels=base_shape[0])

        if x_upsample is not None:
            x_out = alpha * x_upsample + (1 - alpha) * x_out

        return x_out


def from_rgb(x, filters_out, activation, param=None):
    x = conv3d(x, filters_out, 1, activation, param)
    x = tf.nn.leaky_relu(x, alpha=param)
    return x


def discriminator_block(x, filters_in, filters_out, activation, param=None):
    with tf.variable_scope('conv_1'):
        x = conv3d(x, filters_in, 3, activation, param)
        x = tf.nn.leaky_relu(x, alpha=param)
    with tf.variable_scope('conv_2'):
        x = conv3d(x, filters_out, 3, activation, param)
        x = tf.nn.leaky_relu(x, alpha=param)
    x = tf.layers.average_pooling3d(x, 2, 2, data_format='channels_first')
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


def discriminator(x, alpha, phase, num_phases, base_dim, activation, param=None, is_reuse=False):
    with tf.variable_scope('discriminator') as scope:

        if is_reuse:
            scope.reuse_variables()

        x_downscale = x

        with tf.variable_scope(f'from_rgb_{phase}'):
            filters_out = num_filters(phase, num_phases, base_dim)
            x = from_rgb(x, filters_out, activation, param=param)

        for i in reversed(range(2, phase + 1)):

            with tf.variable_scope(f'discriminator_block_{i}'):
                filters_in = num_filters(i, num_phases, base_dim)
                filters_out = num_filters(i - 1, num_phases, base_dim)
                x = discriminator_block(x, filters_in, filters_out, activation, param=param)

            if i == phase:
                with tf.variable_scope(f'from_rgb_{phase - 1}'):
                    fromrgb_prev = from_rgb(
                        tf.layers.average_pooling3d(x_downscale, 2, 2, data_format='channels_first'),
                        filters_out, activation, param=param)

                x = alpha * fromrgb_prev + (1 - alpha) * x

        with tf.variable_scope(f'discriminator_out'):
            # x = minibatch_stddev_layer(x)
            x = conv3d(x, filters_out, 3, activation=activation, param=param)
            x = tf.nn.leaky_relu(x, alpha=param)
            x = tf.layers.flatten(x, data_format='channels_first')
            with tf.variable_scope('dense_1'):
                x = dense(x, base_dim, activation=activation, param=param)
            x = tf.nn.leaky_relu(x, alpha=param)
            with tf.variable_scope('dense_2'):
                x = dense(x, 1, activation='linear')

        return x


def test():
    num_phases = 9
    base_dim = 256
    base_shape = [1, 1, 4, 4]
    for phase in range(8, num_phases):
        shape = [1, 1] + list(np.array(base_shape)[1:] * 2 ** (phase - 1))
        print(shape)
        x = tf.random.normal(shape=shape)
        print('Discriminator output shape:', discriminator(x, 0.5, phase, num_phases, base_dim, activation='leaky_relu', param=0.3).shape)

        for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'):
            print(np.product(p.shape), p.name)  # i.name if you want just a name

        print('Total discriminator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('discriminator')))
        z = tf.random.normal(shape=(1, 256))
        print(generator(z, 0.5, phase, num_phases, base_dim, base_shape, activation='leaky_relu', param=0.3).shape)
        for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'):
            print(np.product(p.shape), p.name)  # i.name if you want just a name

        print('Total generator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('generator')))


if __name__ == '__main__':
    test()
