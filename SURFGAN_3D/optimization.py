import tensorflow as tf

from networks.loss import forward_simultaneous, forward_generator, forward_discriminator

def minimize_with_clipping(optimizer, loss, var_list, clipping):
    """Does minimization by calling compute_gradients and apply_gradients, but optionally clips the gradients in between.
    Parameters:
        optimizer: the optimizer object
        loss: the loss to optimize
        var_list: the variable list of variables that should be updated. To update the generator, e.g. var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        clipping: if clipping should be a applied (bool)
    Returns:
        train_op, gradients, variables, max_norm
        train_op: the training op that can be run in a tf.Session to perform an optimization step
        gradients: the computed gradients
        variables: the variable names corresponding to the gradients
        max_norm: the max_norm of the gradients of this optimization step.
    """

    # Rather then calling *.minimize on the optimizer, we compute gradients as seperate step.
    # This allows gradients to be clipped before they are applied
    gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=var_list))

    if clipping:
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

    # Compute gradient norms for reporting
    norms = tf.stack([tf.norm(grad) for grad in gradients if grad is not None])
    max_norm = tf.reduce_max(norms)

    train_op = optimizer.apply_gradients(zip(gradients, variables))

    return train_op, gradients, variables, max_norm

def optimize_step(optimizer_gen, optimizer_disc, generator, discriminator, real_image_input, latent_dim, alpha, phase,
    num_phases, base_dim, base_shape, activation, leakiness, network_size, loss_fn, gp_weight, optim_strategy, g_clipping = False, d_clipping = False):
    """Defines the op for a single optimization step.
    Parameters:
        optimizer_gen:
        optimizer_disc:
        generator:
        discriminator:
        real_image_input:
        latent_dim:
        alpha:
        phase:
        num_phases:
        base_dim:
        base_shape:
        activation:
        leakiness:
        network_size:
        loss_fn:
        gp_weight:
        optim_strategy:
        g_clipping:
        d_clipping:
    Returns:
        train_gen, train_disc, gen_loss, disc_loss, gp_loss, gen_sample, g_gradients, g_variables, d_gradients, d_variables
        train_gen: generator training op
        train_disc: discriminator training op
        gen_loss: generator loss
        disc_loss: discriminator loss
        gp_loss: gradient penalty component of the loss
        gen_sample: generator samples on which the loss is computed
        g_gradients: generator gradients
        g_variables: generator variables (names)
        d_gradients: discriminator gradients
        d_variables: discriminator variables (names)
    """

    # Perform forward steps of discriminator and generatiour simulatnesouly
    if optim_strategy == 'simultaneous':
        gen_loss, disc_loss, gp_loss, gen_sample = forward_simultaneous(
            generator,
            discriminator,
            real_image_input,
            latent_dim,
            alpha,
            phase,
            num_phases,
            base_dim,
            base_shape,
            activation,
            leakiness,
            network_size,
            loss_fn,
            gp_weight
        )

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        train_gen, g_gradients, g_variables, max_g_norm = minimize_with_clipping(optimizer_gen, gen_loss, gen_vars, g_clipping)

        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        train_disc, d_gradients, d_variables, max_d_norm = minimize_with_clipping(optimizer_disc, disc_loss, disc_vars, d_clipping)

    elif optim_strategy == 'alternate':

        disc_loss, gp_loss = forward_discriminator(
            generator,
            discriminator,
            real_image_input,
            latent_dim,
            alpha,
            phase,
            num_phases,
            base_dim,
            base_shape,
            activation,
            leakiness,
            network_size,
            loss_fn,
            gp_weight,
            # conditioning=real_label
        )

        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        train_disc, d_gradients, d_variables, max_d_norm = minimize_with_clipping(optimizer_disc, disc_loss, disc_vars, d_clipping)

        with tf.control_dependencies([train_disc]):
            gen_sample, gen_loss = forward_generator(
                generator,
                discriminator,
                real_image_input,
                latent_dim,
                alpha,
                phase,
                num_phases,
                base_dim,
                base_shape,
                activation,
                leakiness,
                network_size,
                loss_fn,
                is_reuse=True
            )

            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            train_gen, g_gradients, g_variables, max_g_norm = minimize_with_clipping(optimizer_gen, gen_loss, gen_vars, g_clipping)

    else:
        raise ValueError("Unknown optim strategy ", optim_strategy)

    return train_gen, train_disc, gen_loss, disc_loss, gp_loss, gen_sample, g_gradients, g_variables, d_gradients, d_variables, max_g_norm, max_d_norm