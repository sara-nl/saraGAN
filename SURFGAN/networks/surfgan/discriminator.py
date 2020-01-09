from networks.ops import *
import os
import time


def num_filters(phase, num_phases, base_dim):

    filter_list = [1024, 1024, 256, 256, 256, 128, 64, 64]

    # num_downscales = int(np.log2(base_dim / 64))
    # filters = min(base_dim // (2 ** (phase - num_phases + num_downscales)), base_dim)
    filters = filter_list[phase - 1]
    return filters


def discriminator_block(x, filters_in, filters_out, activation, param=None):
    with tf.variable_scope('conv_1'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x = conv3d(x, filters_out, kernel, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    x = downscale3d(x)
    return x


def discriminator_out(x, latent_dim, activation, param):
    with tf.variable_scope(f'discriminator_out'):
        # x = minibatch_stddev_layer(x)
        # x = conv3d(x, filters_out, 3, activation=activation, param=param)
        # x = apply_bias(x)
        # x = act(x, activation, param=param)
        with tf.variable_scope('dense_1'):
            x = dense(x, latent_dim, activation=activation, param=param)
            x = apply_bias(x)
            x = act(x, activation, param=param)
        with tf.variable_scope('dense_2'):
            x = dense(x, 1, activation='linear')
            x = apply_bias(x)

        return x


def discriminator(x, alpha, phase, num_phases, base_dim, latent_dim, activation, param=None, is_reuse=False):
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

        x = discriminator_out(x, latent_dim, activation, param)
        return x


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = str(16)

    num_phases = 9
    base_dim = 1024
    latent_dim = 1024
    base_shape = [1, 1, 4, 4]
    for phase in range(8, 9):
        shape = [1, 1] + list(np.array(base_shape)[1:] * 2 ** (phase - 1))
        x = tf.random.normal(shape=shape)
        y = discriminator(x, 0.5, phase, num_phases, base_dim, latent_dim, activation='leaky_relu', param=0.3)

        loss = tf.reduce_sum(y)
        optim = tf.train.GradientDescentOptimizer(1e-5)
        train = optim.minimize(loss)
        print('Discriminator output shape:', y.shape)

        for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'):
            print(np.product(p.shape), p.name)  # i.name if you want just a name

        print('Total discriminator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('discriminator')))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            sess.run(train)

            end = time.time()

            print(f"{end - start} seconds")

