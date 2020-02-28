from networks.ops import *
from networks.surfgan.ops import *
import time


def discriminator_block(x, filters_in, filters_out, activation, param=None):
    total_parameters = 0
    with tf.variable_scope('residual') as scope:
        t = downscale3d(x)
        t, runtime_coef = conv3d(t, filters_out, (1, 1, 1), activation, param=param)
        parameters = np.product((1, 1, 1)) * filters_out * filters_in
        total_parameters += parameters

    with tf.variable_scope('conv_1'):
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x, runtime_coef = conv3d(x, filters_in, kernel, activation, param=param)
        x = apply_bias(x, runtime_coef)
        x = act(x, activation, param=param)
        parameters = np.product(kernel) * filters_in * filters_in + filters_in
        total_parameters += parameters

    with tf.variable_scope('conv_2'):

        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        x, runtime_coef = conv3d(x, filters_out, kernel, activation, param=param)
        x = apply_bias(x, runtime_coef)
        x = act(x, activation, param=param)

        parameters = np.product(kernel) * filters_out * filters_in + filters_out
        total_parameters += parameters

    x = downscale3d(x)
    print(f'\tOutput Shape: {x.shape}\tParameters: {total_parameters}\t Kernel: {kernel}')
    x = (x + t) * (1 / calculate_gain(activation, param))

    return x


def discriminator_out(x, base_dim, latent_dim, filters_out, activation, param, conditioning=None):
    with tf.variable_scope(f'discriminator_out'):
        x = minibatch_stddev_layer(x)
        with tf.variable_scope('conv1') as scope:
            shape = x.get_shape().as_list()[2:]
            kernel = [k(s) for s in shape]
            filters_in = x.shape[1]
            x, runtime_coef = conv3d(x, filters_out, kernel, activation=activation, param=param)
            x = apply_bias(x, runtime_coef)
            x = act(x, activation, param=param)
            print(f'{scope.name}\n\tOutput Shape: {x.shape}\tParameters: {np.product(kernel) * filters_out * filters_in + filters_out}\tKernel: {kernel}')

        with tf.variable_scope('dense1') as scope:
            filters_in = np.prod(x.shape[1:])
            filters_out = latent_dim
            x, runtime_coef = dense(x, filters_out, activation=activation, param=param)
            x = apply_bias(x, runtime_coef)
            x = act(x, activation, param=param)
            print(f'{scope.name}\n\tOutput Shape: {x.shape}\tParameters: {filters_out * filters_in + filters_out}')

        with tf.variable_scope('dense2') as scope:
            filters_in = np.prod(x.shape[1:])
            filters_out = 1
            x, runtime_coef = dense(x, filters_out, activation='linear')
            print(f'{scope.name}\n\tOutput Shape: {x.shape}\tParameters: {filters_out * filters_in + filters_out}')
            x = apply_bias(x, runtime_coef)

            if conditioning is not None:
                x = tf.reduce_sum(x * conditioning, axis=1, keepdims=True)

        return x


def discriminator(x, alpha, phase, num_phases, base_dim, latent_dim, activation, conditioning=None, param=None, is_reuse=False, size='m', ):

    with tf.variable_scope('discriminator') as scope:

        if is_reuse:
            scope.reuse_variables()

        x_downscale = x

        with tf.variable_scope(f'from_rgb_{phase}') as scope:
            filters_out = num_filters(phase, num_phases, base_dim, size=size)
            filters_in = x.shape[1]
            x = from_rgb(x, filters_out, activation, param=param)
            print(f'{scope.name}\n\tOutput Shape: {x.shape}\tParameters: {1*1*1*filters_out * filters_in + filters_out}\tKernel: (1, 1, 1)')

        for i in reversed(range(2, phase + 1)):
            with tf.variable_scope(f'discriminator_block_{i}') as scope:
                print(scope.name)
                filters_in = num_filters(i, num_phases, base_dim, size=size)
                filters_out = num_filters(i - 1, num_phases, base_dim, size=size)
                x = discriminator_block(x, filters_in, filters_out, activation, param=param)
            if i == phase:
                with tf.variable_scope(f'from_rgb_{phase - 1}'):
                    fromrgb_prev = from_rgb(
                        downscale3d(x_downscale),
                        filters_out, activation, param=param)

                x = alpha * fromrgb_prev + (1 - alpha) * x


        x = discriminator_out(x, base_dim, latent_dim, filters_out, activation, param, conditioning)

        return x


def print_network(network):

    network_dict = defaultdict(dict)

    seen = []

    all_lines = []

    for j, p in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=network)):

        hierarchy = p.name.split('/')
        scope_cascade = ''

        for i, scope in enumerate(hierarchy):
            scope_full = scope_cascade + scope
            if scope_full in seen:
                scope_cascade += f'{scope}/'
                continue
            else:
                num_parameters_in_scope = str(sum(np.product(p.shape) for p in tf.trainable_variables(scope_full)))
                whitespace_pre = ' ' * (len(scope_full) - len(scope))
                whitespace_mid = ' ' * (100 - len(num_parameters_in_scope) - len(scope) - len(whitespace_pre))
                line = whitespace_pre + scope + whitespace_mid + num_parameters_in_scope
                all_lines.append(line)
                scope_cascade += f'{scope}/'
                seen.append(scope_full)

        # if j == 5:
        #     break

    for i, line in enumerate(all_lines):
        print(line)








if __name__ == '__main__':
    num_phases = 8
    latent_dim = 512
    base_shape = [1, 1, 4, 4]
    size = 'm'
    base_dim = num_filters(-num_phases + 1, num_phases, base_dim=None, size=size)
    for phase in range(8, 9):
        tf.reset_default_graph()
        shape = [1, 1] + list(np.array(base_shape)[1:] * 2 ** (phase - 1))
        x = tf.random.normal(shape=shape)
        y = discriminator(x, 0.5, phase, num_phases, base_dim, latent_dim, activation='leaky_relu', param=0.3)

        loss = tf.reduce_sum(y)
        optim = tf.train.GradientDescentOptimizer(1e-5)
        train = optim.minimize(loss)

        # print_network('discriminator')
#         print('Discriminator output shape:', y.shape)
#
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

