from networks.ops import *
import time


def generator_in(x, filters, shape, activation, param=None):

    with tf.variable_scope('dense'):
        x = dense(x, np.product(shape) * filters, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    x = tf.reshape(x, [-1, filters] + list(shape))

    with tf.variable_scope('conv'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]

        x = conv3d(x, filters, kernel, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        x = pixel_norm(x)
    return x


def generator_block(x, filters_out, activation, param=None):

    with tf.variable_scope('residual'):
        t = conv3d(x, filters_out, (1, 1, 1), activation=activation, param=param)
        t = upscale3d(t)

    with tf.variable_scope('upsample'):
        x = upscale3d(x)

    with tf.variable_scope('conv_1'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x = conv3d(x, filters_out, kernel, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        x = pixel_norm(x)

    with tf.variable_scope('conv_2'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x = conv3d(x, filters_out, kernel, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        x = pixel_norm(x)

    x = (x + t) * (1 / calculate_gain(activation, param))

    return x


def generator(x, alpha, phase, num_phases, base_dim, base_shape, activation, param=None, size='m', is_reuse=False, conditioning=None):

    if conditioning is not None:
        raise NotImplementedError()

    channels = base_shape[0]

    with tf.variable_scope('generator') as scope:

        if is_reuse:
            scope.reuse_variables()

        with tf.variable_scope('generator_in'):
            x = generator_in(x, filters=base_dim, shape=base_shape[1:], activation=activation, param=param)

        with tf.variable_scope(f'to_rgb_{1}'):
            x_out = to_rgb(x, channels)

        for i in range(2, phase + 1):

            filters_out = num_filters(i, num_phases, base_shape, base_dim, size=size)
            with tf.variable_scope(f'generator_block_{i}'):
                x = generator_block(x, filters_out, activation=activation, param=param)

            if i == phase:
                with tf.variable_scope(f'to_rgb_{i}'):
                    x_out = (1 - alpha) * to_rgb(x) + upscale3d(x_out)
            else:
                with tf.variable_scope(f'to_rgb_{i}'):
                    x_out = to_rgb(x) + upscale3d(x_out)

        return x_out


if __name__ == '__main__':
    num_phases = 8
    latent_dim = 512
    base_shape = [1, 1, 4, 4]
    base_dim = num_filters(-num_phases + 1, num_phases, base_shape, base_dim=None, size='m')
    for phase in range(3, 4):
        shape = [1, latent_dim]
        x = tf.random.normal(shape=shape)
        y = generator(x, 0.5, phase, num_phases, base_dim, base_shape, activation='leaky_relu',
                      param=0.3)

        loss = tf.reduce_sum(y)
        optim = tf.train.GradientDescentOptimizer(1e-5)
        train = optim.minimize(loss)
        print('Generator output shape:', y.shape)

        for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'):
            print(np.product(p.shape), p.name)  # i.name if you want just a name

        print('Total generator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('generator')))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            sess.run(train)

            end = time.time()

            print(f"{end - start} seconds")
