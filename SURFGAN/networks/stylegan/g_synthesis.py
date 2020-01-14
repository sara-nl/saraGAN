from networks.ops import *

def generator_in(d_z, noise_inputs, base_dim, base_shape, activation, param=None):

    with tf.variable_scope('constant_in'):
        x = tf.get_variable('input_constant',
                            shape=[1, base_dim, *base_shape[1:]],
                            initializer=tf.initializers.ones())

        x = tf.tile(x, [tf.shape(d_z)[0], 1, 1, 1, 1])
        x = apply_noise(x, noise_inputs[0], randomize_noise=True)
        x = apply_bias(x)
        x = act(x, activation, param)
        x = instance_norm(x)
        x = style_mod(x, d_z[:, 0], activation, param)

    with tf.variable_scope('conv'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x = conv3d(x, base_dim, kernel, activation, param)
        x = apply_noise(x, noise_inputs[0], randomize_noise=True)
        x = apply_bias(x)
        x = act(x, activation, param)
        x = instance_norm(x)
        x = style_mod(x, d_z[:, 0], activation, param)

    return x


def generator_block(x, filters_out, d_z, noise_inputs, layer_idx, activation, param=None):
    with tf.variable_scope('upsample'):
        x = upscale3d(x)

    with tf.variable_scope('conv_1'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x = conv3d(x, filters_out, kernel, activation, param)
        x = apply_noise(x, noise_inputs[layer_idx - 1], randomize_noise=True)
        x = apply_bias(x)
        x = act(x, activation, param)
        x = instance_norm(x)
        x = style_mod(x, d_z[:, layer_idx - 1], activation, param)

    with tf.variable_scope('conv_2'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x = conv3d(x, filters_out, kernel, activation, param)
        x = apply_noise(x, noise_inputs[layer_idx - 1], randomize_noise=True)
        x = apply_bias(x)
        x = act(x, activation, param)
        x = instance_norm(x)
        x = style_mod(x, d_z[:, layer_idx - 1], activation, param)
    return x


def g_synthesis(d_z,
                alpha,
                phase,
                num_phases,
                base_dim,
                base_shape,
                activation,
                param=None):

    noise_inputs = list()
    for layer_idx in range(1, phase + 1):
        shape = [1, 1] + list(np.array(base_shape[1:]) * 2 ** (layer_idx - 1))
        noise_inputs.append(tf.get_variable(f'noise_input_{layer_idx}', shape=shape,
                                            initializer=tf.initializers.random_normal(), trainable=False))

    with tf.variable_scope('g_synthesis'):

        with tf.variable_scope('generator_in'):
            x = generator_in(d_z, noise_inputs, base_dim, base_shape, activation, param)

        x_upsample = None

        for layer_idx in range(2, phase + 1):

            if layer_idx == phase:
                with tf.variable_scope(f'to_rgb_{phase - 1}'):
                    x_upsample = upscale3d(to_rgb(x, channels=base_shape[0]))

            filters_out = num_filters(layer_idx, num_phases, base_dim)
            with tf.variable_scope(f'generator_block_{layer_idx}'):
                x = generator_block(x, filters_out, d_z, noise_inputs, layer_idx, activation=activation,
                                    param=param)

        with tf.variable_scope(f'to_rgb_{phase}'):
            x_out = to_rgb(x, channels=base_shape[0])

        if x_upsample is not None:
            x_out = alpha * x_upsample + (1 - alpha) * x_out

        return x_out


if __name__ == '__main__':

    num_phases = 8
    base_dim = 1024
    latent_size = 1024
    base_shape = (1, 1, 4, 4)

    for phase in range(4, 5):
        tf.reset_default_graph()
        latents_shape = (1, phase, latent_size)
        dlatents = tf.random.normal(shape=latents_shape)
        alpha = tf.Variable(0.5)
        img_out = g_synthesis(dlatents, alpha, phase, num_phases, base_dim, base_shape, 'leaky_relu', 0.2)
        print('Total synthesis generator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('g_synthesis')))

        print('Output shape:', img_out.shape)
