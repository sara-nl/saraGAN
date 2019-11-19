from networks.ops import *


def discriminator_block(x, filters_in, filters_out, activation, param=None):
    with tf.variable_scope('conv_1'):
        x = conv3d(x, filters_in, 3, activation, param)
        x = apply_bias(x)
        x = act(x, activation, param)
    with tf.variable_scope('conv_2'):
        x = conv3d(x, filters_out, 3, activation, param)
        x = apply_bias(x)
        x = act(x, activation, param)
    x = downscale3d(x)
    return x


def discriminator(x, alpha, phase, num_phases, base_dim, activation, param=None, is_reuse=False):
    with tf.variable_scope('discriminator') as scope:

        if is_reuse:
            scope.reuse_variables()

        x_downscale = x

        with tf.variable_scope(f'from_rgb_{phase}'):
            filters_out = num_filters(phase, num_phases, base_dim)
            x = from_rgb(x, filters_out, activation, param=param)

        for layer_idx in reversed(range(2, phase + 1)):

            with tf.variable_scope(f'discriminator_block_{layer_idx}'):
                filters_in = num_filters(layer_idx, num_phases, base_dim)
                filters_out = num_filters(layer_idx - 1, num_phases, base_dim)
                x = discriminator_block(x, filters_in, filters_out, activation, param=param)

            if layer_idx == phase:
                with tf.variable_scope(f'from_rgb_{phase - 1}'):
                    fromrgb_prev = from_rgb(downscale3d(x_downscale), filters_out, activation, param=param)

                x = alpha * fromrgb_prev + (1 - alpha) * x

        with tf.variable_scope(f'discriminator_out'):
            # x = minibatch_stddev_layer(x)
            x = conv3d(x, filters_out, 3, activation=activation, param=param)
            x = apply_bias(x)
            x = act(x, activation, param)
            with tf.variable_scope('dense_1'):
                x = dense(x, base_dim, activation=activation, param=param)
                x = apply_bias(x)
                x = act(x, activation, param)
            with tf.variable_scope('dense_2'):
                x = dense(x, 1, activation='linear')
                x = apply_bias(x)

        return x


if __name__ == '__main__':

    num_phases = 8
    base_dim = 256
    base_shape = [1, 1, 4, 4]
    for phase in range(1, num_phases + 1):
        tf.reset_default_graph()
        shape = [1, 1] + list(np.array(base_shape)[1:] * 2 ** (phase - 1))
        print(shape)
        x = tf.random.normal(shape=shape)
        print('Discriminator output shape:', discriminator(x, 0.5, phase, num_phases, base_dim, activation='leaky_relu', param=0.3).shape)

        # for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'):
        #     print(np.product(p.shape), p.name)  # i.name if you want just a name

        print('Total discriminator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('discriminator')))
