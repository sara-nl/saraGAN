from networks.ops import *
from networks.surfgan.ops import *  # No good coding style, but ensures to use these ops.
import time


def discriminator_block(x, filters_in, filters_out, activation, param=None):

    with tf.variable_scope('residual'):
        t, runtime_coef = conv2d(x, filters_out, (1, 1), activation, param=param)
        t = downscale2d(t)

    with tf.variable_scope('conv_1'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x, runtime_coef = conv2d(x, filters_in, kernel, activation, param=param)
        x = apply_bias(x, runtime_coef)
        x = act(x, activation, param=param)
    with tf.variable_scope('conv_2'):

        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x, runtime_coef = conv2d(x, filters_out, kernel, activation, param=param)
        x = apply_bias(x, runtime_coef)
        x = act(x, activation, param=param)

    x = downscale2d(x)
    x = (x + t) * (1 / calculate_gain(activation, param))

    return x


def discriminator_out(x, base_dim, latent_dim, filters_out, activation, param):
    with tf.variable_scope(f'discriminator_out'):
        # x = minibatch_stddev_layer(x)
        with tf.variable_scope('conv'):
            shape = x.get_shape().as_list()[2:]
            kernel = [k(s) for s in shape]
            x, runtime_coef = conv2d(x, filters_out, kernel, activation=activation, param=param)
            x = apply_bias(x, runtime_coef)
            x = act(x, activation, param=param)
        with tf.variable_scope('dense'):
            x, runtime_coef = dense(x, 1, activation='linear')
            x = apply_bias(x, runtime_coef)

        return x


def discriminator(x, alpha, phase, num_phases, base_dim, latent_dim, activation, param=None, is_reuse=False, size='medium'):

    with tf.variable_scope('discriminator') as scope:

        if is_reuse:
            scope.reuse_variables()

        x_downscale = x

        with tf.variable_scope(f'from_rgb_{phase}'):
            filters_out = num_filters(phase, num_phases, base_dim, size=size)
            x = from_rgb(x, filters_out, activation, param=param)

        for i in reversed(range(2, phase + 1)):
            with tf.variable_scope(f'discriminator_block_{i}'):
                filters_in = num_filters(i, num_phases, base_dim, size=size)
                filters_out = num_filters(i - 1, num_phases, base_dim, size=size)
                x = discriminator_block(x, filters_in, filters_out, activation, param=param)

            if i == phase:
                with tf.variable_scope(f'from_rgb_{phase - 1}'):
                    fromrgb_prev = from_rgb(
                        downscale2d(x_downscale),
                        filters_out, activation, param=param)

                x = alpha * fromrgb_prev + (1 - alpha) * x

        x = discriminator_out(x, base_dim, latent_dim, filters_out, activation, param)
        return x


if __name__ == '__main__':
    num_phases = 7
    base_dim = 512
    base_shape = [3, 4, 4]
    latent_dim = 512
    for phase in range(7, 8):
        tf.reset_default_graph()
        shape = [1, 3] + list(np.array(base_shape)[1:] * 2 ** (phase - 1))
        print(shape)
        x = tf.random.normal(shape=shape)
        y = discriminator(x, 0.5, phase, num_phases, base_dim, latent_dim, activation='leaky_relu', param=0.3)

        loss = tf.reduce_sum(y)
        optim = tf.train.GradientDescentOptimizer(1e-5)
        train = optim.minimize(loss)
        print('Discriminator output shape:', y.shape)

        for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'):
            print(np.product(p.shape), p.name)  # i.name if you want just a name

        print('Total discriminator parameters:',
              sum(np.product(p.shape) for p in tf.trainable_variables('discriminator')))

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     start = time.time()
        #     sess.run(train)

        #     end = time.time()

        #     print(f"{end - start} seconds")

