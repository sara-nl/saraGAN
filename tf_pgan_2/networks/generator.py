from networks.ops import *


def generator_in(x, filters, shape, activation, param=None):
    with tf.variable_scope('dense'):
        x = dense(x, np.product(shape) * filters, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    x = tf.reshape(x, [-1, filters] + list(shape))
    with tf.variable_scope('conv'):
        x = conv3d(x, filters, 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        x = pixel_norm(x)
    return x


def generator_block(x, filters_out, activation, param=None):
    with tf.variable_scope('upsample'):
        x = upscale3d(x)

    with tf.variable_scope('conv_1'):
        x = conv3d(x, filters_out, 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        x = pixel_norm(x)

    with tf.variable_scope('conv_2'):
        x = conv3d(x, filters_out, 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        x = pixel_norm(x)
    return x


def generator(x, alpha, phase, num_phases, base_dim, base_shape, activation, param=None):
    with tf.variable_scope('generator'):
        with tf.variable_scope('generator_in'):
            x = generator_in(x, filters=base_dim, shape=base_shape[1:], activation=activation, param=param)

        x_upsample = None

        for i in range(2, phase + 1):

            if i == phase:
                with tf.variable_scope(f'to_rgb_{phase - 1}'):
                    x_upsample = upscale3d(to_rgb(x, channels=base_shape[0]))

            filters_out = num_filters(i, num_phases, base_dim)
            with tf.variable_scope(f'generator_block_{i}'):
                x = generator_block(x, filters_out, activation=activation, param=param)

        with tf.variable_scope(f'to_rgb_{phase}'):
            x_out = to_rgb(x, channels=base_shape[0])

        if x_upsample is not None:
            x_out = alpha * x_upsample + (1 - alpha) * x_out

        return x_out