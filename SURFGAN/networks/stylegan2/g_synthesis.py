# pylint: disable=unused-wildcard-import
from networks.ops import *
from networks.stylegan2.ops import modulated_conv3d, to_rgb


def num_filters(phase, num_phases, base_dim):
    filter_list = [512, 512, 128, 128, 128, 64, 32, 16]
    assert num_phases == len(filter_list)
    filters = filter_list[phase - 1]
    return filters


def generator_in(d_z, base_dim, base_shape, activation, param=None):

    with tf.variable_scope('constant_in'):
        x = tf.get_variable('input_constant',
                            shape=[1, base_dim, *base_shape[1:]],
                            initializer=tf.initializers.random_normal())

        x = tf.tile(x, [tf.shape(d_z)[0], 1, 1, 1, 1])

    with tf.variable_scope('conv'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x = modulated_conv3d(x, d_z[:, 0], base_dim, kernel, activation=activation, param=param)
        x = apply_noise(x)
        x = apply_bias(x)
        x = act(x, activation, param)
    return x


def generator_block(x, filters_out, d_z, layer_idx, activation, param=None):

    with tf.variable_scope('conv_1'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x = modulated_conv3d(x, d_z[:, layer_idx * 3 - 5], filters_out, kernel, activation=activation, up=True, param=param)
        x = apply_noise(x)
        x = apply_bias(x)
        x = act(x, activation, param)

    with tf.variable_scope('conv_2'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x = modulated_conv3d(x, d_z[:, layer_idx * 3 - 4], filters_out, kernel, activation=activation, param=param)
        x = apply_noise(x)
        x = apply_bias(x)
        x = act(x, activation, param)
    return x


def g_synthesis(d_z,
                alpha,
                phase,
                num_phases,
                base_dim,
                base_shape,
                activation,
                param=None):

    with tf.variable_scope('g_synthesis'):

        with tf.variable_scope('generator_in'):
            x = generator_in(d_z, base_dim, base_shape, activation, param)

        x_out = to_rgb(x, d_z[:, 0])

        for layer_idx in range(2, phase + 1):

            filters_out = num_filters(layer_idx, num_phases, base_dim)
            with tf.variable_scope(f'generator_block_{layer_idx}'):
                x = generator_block(x, filters_out, d_z, layer_idx, activation=activation,
                                    param=param)

            with tf.variable_scope(f'to_rgb_{layer_idx}'):
                x_out = to_rgb(x, d_z[:, layer_idx * 3 - 3]) + upscale3d(x_out)

        return x_out


if __name__ == '__main__':

    num_phases = 8
    base_dim = 1024
    latent_size = 1024
    base_shape = (1, 1, 4, 4)

    for phase in range(1, 2):
        tf.reset_default_graph()
        latents_shape = (1, phase * 3 - 2, latent_size)
        dlatents = tf.random.normal(shape=latents_shape)
        alpha = tf.Variable(0.5)
        img_out = g_synthesis(dlatents, alpha, phase, num_phases,
                              base_dim, base_shape, 'leaky_relu', 0.2)
        for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope='g_synthesis'):
            # i.name if you want just a name
            print(np.product(p.shape), p.name)

        print('Total synthesis generator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables(
                  'g_synthesis')))

        print('Output shape:', img_out.shape)
