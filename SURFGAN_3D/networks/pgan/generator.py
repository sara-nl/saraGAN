from networks.ops import *
import time

def get_filters_generator(filter_spec, phase_i, layer_i):
    if phase_i >= len(filter_spec):
        print(f"Error: no filter count specified for phase {phase_i}. Please check the file passed to --filter_spec.")
        raise ValueError
    if layer_i >= len(filter_spec[phase_i]):
        print(f"Error: no filter count specified for layer {layer_i} in phase {phase_i}. Please check the file passed to --filter_spec.")
        raise ValueError
    # DEBUG:
    #print(f"Returning filter_spec[{phase_i}][{layer_i}] = {filter_spec[phase_i][layer_i]}")
    return filter_spec[phase_i][layer_i]

def get_kernels_generator(kernel_spec, phase_i, layer_i):
    if phase_i >= len(kernel_spec):
        print(f"Error: no kernel shape specified for phase {phase_i}. Please check the file passed to --kernel_spec.")
        raise ValueError
    if layer_i >= len(kernel_spec[phase_i]):
        print(f"Error: no kernel shape specified for layer {layer_i} in phase {phase_i}. Please check the file passed to --kernel_spec.")
        raise ValueError
    # DEBUG:
    #print(f"Returning kernel_spec[{phase_i}][{layer_i}] = {kernel_spec[phase_i][layer_i]}")
    return kernel_spec[phase_i][layer_i]

def generator_in(x, filters, shape, activation, kernel_shape, kernel_spec, filter_spec, param=None):

    with tf.variable_scope('dense'):
        # x = dense(x, np.product(shape) * filters, activation, param=param)
        x = dense(x, np.product(shape) * get_filters_generator(filter_spec, 0, 0), activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    print(f"Reshaping to: {[-1, filters] + list(shape)}")
    # x = tf.reshape(x, [-1, filters] + list(shape))
    x = tf.reshape(x, [-1, get_filters_generator(filter_spec, 0, 0)] + list(shape))

    with tf.variable_scope('conv'):
        shape = x.get_shape().as_list()[2:]
        kernel = get_kernel(shape, kernel_shape)

        x = conv3d(x, get_filters_generator(filter_spec, 0, 1), get_kernels_generator(kernel_spec, 0, 1), activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        x = pixel_norm(x)
    return x


def generator_block(x, filters_out, activation, kernel_shape, kernel_spec, filter_spec, i, param=None):
    with tf.variable_scope('upsample'):
        x = upscale3d(x)

    with tf.variable_scope('conv_1'):
        # shape = x.get_shape().as_list()[2:]
        #kernel = get_kernel(shape, kernel_shape)
        kernel = get_kernels_generator(kernel_spec, i-1, 0)
        #x = conv3d(x, filters_out, kernel, activation, param=param)
        x = conv3d(x, get_filters_generator(filter_spec, i-1, 0), kernel, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        x = pixel_norm(x)

    with tf.variable_scope('conv_2'):
        # shape = x.get_shape().as_list()[2:]
        #kernel = get_kernel(shape, kernel_shape)
        kernel = get_kernels_generator(kernel_spec, i-1, 1)
        # x = conv3d(x, filters_out, kernel, activation, param=param)
        x = conv3d(x, get_filters_generator(filter_spec, i-1, 1), kernel, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
        x = pixel_norm(x)
    return x


def generator(x, alpha, phase, num_phases, base_dim, base_shape, activation, kernel_shape, kernel_spec, filter_spec, param=None, size='medium', is_reuse=False, conditioning=None):

    if conditioning is not None:
        raise NotImplementedError()

    with tf.variable_scope('generator') as scope:

        if is_reuse:
            scope.reuse_variables()
        with tf.variable_scope('generator_in'):
            x = generator_in(x, filters=base_dim, shape=base_shape[1:], activation=activation, kernel_shape=kernel_shape, kernel_spec=kernel_spec, filter_spec=filter_spec, param=param)

        x_upsample = None

        for i in range(2, phase + 1):

            if i == phase:
                with tf.variable_scope(f'to_rgb_{phase - 1}'):
                    x_upsample = upscale3d(to_rgb(x, channels=base_shape[0]))
            filters_out = num_filters(i, num_phases, base_shape, base_dim, size=size)
            with tf.variable_scope(f'generator_block_{i}'):
                x = generator_block(x, filters_out, activation=activation, kernel_shape=kernel_shape, kernel_spec=kernel_spec, filter_spec=filter_spec, i=i, param=param)

        with tf.variable_scope(f'to_rgb_{phase}'):
            x_out = to_rgb(x, channels=base_shape[0])

        if x_upsample is not None:
            x_out = alpha * x_upsample + (1 - alpha) * x_out

        return x_out


if __name__ == '__main__':
    num_phases = 8
    base_dim = 1024
    latent_dim = 1024
    base_shape = [1, 1, 4, 4]
    for phase in range(8, 9):
        shape = [1, latent_dim]
        x = tf.random.normal(shape=shape)
        y = generator(x, 0.5, phase, num_phases, base_dim, base_shape, kernel_shape = [3, 3, 3], activation='leaky_relu',
                      param=0.3)

        loss = tf.reduce_sum(y)
        optim = tf.train.GradientDescentOptimizer(1e-5)
        train = optim.minimize(loss)
        print('Generator output shape:', y.shape)

        for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'):
            print(np.product(p.shape), p.name)  # i.name if you want just a name

        print('Total generator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('generator')))

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     start = time.time()
        #     sess.run(train)

        #     end = time.time()

        #     print(f"{end - start} seconds")
