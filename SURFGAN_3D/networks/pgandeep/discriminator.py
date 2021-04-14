from networks.ops import *

def get_filters_discriminator(filter_spec, phase_i, layer_i):
    if phase_i >= len(filter_spec):
        print(f"Error: no filter count specified for phase {phase_i}. Please check the file passed to --filter_spec.")
        raise ValueError
    if layer_i >= len(filter_spec[phase_i]):
        print(f"Error: no filter count specified for layer {layer_i} in phase {phase_i}. Please check the file passed to --filter_spec.")
        raise ValueError
    # DEBUG:
    # print(f"Returning filter_spec[{phase_i}][{layer_i}] = {filter_spec[phase_i][layer_i]}")
    return filter_spec[phase_i][layer_i]

def get_kernels_discriminator(kernel_spec, phase_i, layer_i):
    if phase_i >= len(kernel_spec):
        print(f"Error: no kernel shape specified for phase {phase_i}. Please check the file passed to --kernel_spec.")
        raise ValueError
    if layer_i >= len(kernel_spec[phase_i]):
        print(f"Error: no kernel shape specified for layer {layer_i} in phase {phase_i}. Please check the file passed to --kernel_spec.")
        raise ValueError
    # DEBUG:
    # print(f"Returning kernel_spec[{phase_i}][{layer_i}] = {kernel_spec[phase_i][layer_i]}")
    return kernel_spec[phase_i][layer_i]

def discriminator_block(x, activation, kernel_spec, filter_spec, i, param=None):

    num_layers=len(kernel_spec[i-1])
    print(f'Discriminator_block {i-1}: num_layers={num_layers}')
    for layer_i in range(1, num_layers+1):
        with tf.variable_scope(f'conv_{layer_i}'):
            kernel = get_kernels_discriminator(kernel_spec, i-1, 1)
            # for last layer, take the last filter spec from the previous phase
            if layer_i == num_layers:
                filters = get_filters_discriminator(filter_spec, i-2, num_layers-1)
            else:
                filters = get_filters_discriminator(filter_spec, i-1, num_layers-layer_i-1)
            x = conv3d(x, filters, kernel, activation, param=param)
            x = apply_bias(x)
            x = act(x, activation, param=param)

    # with tf.variable_scope('conv_1'):
    #     # shape = x.get_shape().as_list()[2:]
    #     # kernel = get_kernel(shape, kernel_shape)
    #     kernel = get_kernels_discriminator(kernel_spec, i-1, 1)
    #     # x = conv3d(x, filters_in, kernel, activation, param=param)
    #     x = conv3d(x, get_filters_discriminator(filter_spec, i-1, 0), kernel, activation, param=param)
    #     x = apply_bias(x)
    #     x = act(x, activation, param=param)
    # with tf.variable_scope('conv_2'):

    #     # shape = x.get_shape().as_list()[2:]
    #     # kernel = get_kernel(shape, kernel_shape)
    #     kernel = get_kernels_discriminator(kernel_spec, i-1, 0)
    #     # x = conv3d(x, filters_out, kernel, activation, param=param)
    #     x = conv3d(x, get_filters_discriminator(filter_spec, i-2, 1), kernel, activation, param=param)
    #     x = apply_bias(x)
    #     x = act(x, activation, param=param)
    x = downscale3d(x)
    return x


def discriminator_out(x, latent_dim, activation, kernel_spec, filter_spec, param):
    with tf.variable_scope(f'discriminator_out'):
        num_layers=len(kernel_spec[0])
        print(f'Discriminator_out: num_layers={num_layers}')
        for layer_i in range(1, num_layers): # one less than in discriminator_block, since we have the dense layers here
            with tf.variable_scope(f'conv_{layer_i}'):
                kernel = get_kernels_discriminator(kernel_spec, 0, num_layers - layer_i)
                filters = get_filters_discriminator(filter_spec, 0, num_layers - layer_i - 1)
                x = conv3d(x, filters, kernel, activation=activation, param=param)
                x = apply_bias(x)
                x = act(x, activation, param=param)

#         # x = minibatch_stddev_layer(x)
#         # shape = x.get_shape().as_list()[2:]
#         # kernel = get_kernel(shape, kernel_shape)
#         kernel = get_kernels_discriminator(kernel_spec, 0, 1)
# #        x = conv3d(x, filters_out, kernel, activation=activation, param=param)
#         # base_dim = num_filters for the first generator layer after the latent space. Discriminator should mirror that.
#         # x = conv3d(x, base_dim, kernel, activation=activation, param=param)
#         x = conv3d(x, get_filters_discriminator(filter_spec, 0, 0), kernel, activation=activation, param=param)
#         x = apply_bias(x)
#         x = act(x, activation, param=param)
        with tf.variable_scope('dense_1'):
            x = dense(x, latent_dim, activation=activation, param=param)
            x = apply_bias(x)
            x = act(x, activation, param=param)
        with tf.variable_scope('dense_2'):
            x = dense(x, 1, activation='linear')
            x = apply_bias(x)

        return x


def discriminator(x, alpha, phase, latent_dim, activation, kernel_spec, filter_spec, param=None, is_reuse=False, conditioning=None):

    if conditioning is not None:
        raise NotImplementedError()

    with tf.variable_scope('discriminator') as scope:

        if is_reuse:
            scope.reuse_variables()

        x_downscale = x

        with tf.variable_scope(f'from_rgb_{phase}'):
            # filters_out = num_filters(phase, num_phases, base_shape, base_dim, size=size)
            # x = from_rgb(x, filters_out, activation, param=param)
            # print(f"filters_out in from_rgb_{phase}: {filters_out}")
            x = from_rgb(x, get_filters_discriminator(filter_spec, phase-1, 1), activation, param=param)

        for i in reversed(range(2, phase + 1)):

            with tf.variable_scope(f'discriminator_block_{i}'):
                #filters_in = num_filters(i, num_phases, base_shape, base_dim, size=size)
                #filters_out = num_filters(i - 1, num_phases, base_shape, base_dim, size=size)
                # print(f"Phase {i} filters_in {filters_in} filters_out {filters_out}")
                x = discriminator_block(x, activation, kernel_spec, filter_spec, i=i, param=param)

            if i == phase:
                with tf.variable_scope(f'from_rgb_{phase - 1}'):
                    # print(f"Filters_out: {filters_out}")
                    fromrgb_prev = from_rgb(
                        downscale3d(x_downscale),
                        #filters_out, activation, param=param)
                        get_filters_discriminator(filter_spec, phase-2, 1), activation, param=param)

                x = alpha * fromrgb_prev + (1 - alpha) * x

        x = discriminator_out(x, latent_dim, activation, kernel_spec, filter_spec, param)
        return x


if __name__ == '__main__':
    num_phases = 8
    base_dim = 1024
    base_shape = [1, 1, 4, 4]
    latent_dim = 1024
    for phase in range(4, 5):
        tf.reset_default_graph()
        shape = [1, 1] + list(np.array(base_shape)[1:] * 2 ** (phase - 1))
        print(shape)
        x = tf.random.normal(shape=shape)
        y = discriminator(x, 0.5, phase, num_phases, base_dim, latent_dim, activation='leaky_relu', kernel_shape = [3, 3, 3], param=0.3)

        loss = tf.reduce_sum(y)
        optim = tf.train.GradientDescentOptimizer(1e-5)
        train = optim.minimize(loss)
        print('Discriminator output shape:', y.shape)

        for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'):
            print(np.product(p.shape), p.name)  # i.name if you want just a name

        print('Total discriminator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('discriminator')))

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     start = time.time()
        #     sess.run(train)

        #     end = time.time()

        #     print(f"{end - start} seconds")

