from networks.ops import *


def discriminator_block(x, filters_in, filters_out, activation, param=None):
    with tf.variable_scope('conv_1'):
        x = conv3d(x, filters_in, 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    with tf.variable_scope('conv_2'):
        x = conv3d(x, filters_out, 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    x = downscale3d(x)
    return x


def discriminator_out(x, base_dim, filters_out, activation, param):
    with tf.variable_scope(f'discriminator_out'):
        # x = minibatch_stddev_layer(x)
        x = conv3d(x, filters_out, 3, activation=activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        with tf.variable_scope('dense_1'):
            x = dense(x, base_dim, activation=activation, param=param)
            x = apply_bias(x)
            x = act(x, activation, param=param)
        with tf.variable_scope('dense_2'):
            x = dense(x, 1, activation='linear')
            x = apply_bias(x)

        return x


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
                        downscale3d(x_downscale),
                        filters_out, activation, param=param)

                x = alpha * fromrgb_prev + (1 - alpha) * x

        x = discriminator_out(x, base_dim, filters_out, activation, param)
        return x
