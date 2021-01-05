import tensorflow as tf
import numpy as np

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
    num_phases, base_dim, base_shape, activation, leakiness, network_size, loss_fn, gp_weight, optim_strategy, g_clipping, d_clipping, noise_stddev):
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
        noise_stddev:
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

    # Perform forward steps of discriminator and generator simultaneously
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
            gp_weight,
            noise_stddev
        )

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        train_gen, g_gradients, g_variables, max_g_norm = minimize_with_clipping(optimizer_gen, gen_loss, gen_vars, g_clipping)

        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        train_disc, d_gradients, d_variables, max_d_norm = minimize_with_clipping(optimizer_disc, disc_loss, disc_vars, d_clipping)

    # Perform forward steps of discriminator and generator alternatingly
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
            noise_stddev,
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
                noise_stddev,
                is_reuse=True
            )

            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            train_gen, g_gradients, g_variables, max_g_norm = minimize_with_clipping(optimizer_gen, gen_loss, gen_vars, g_clipping)

    else:
        raise ValueError("Unknown optim strategy ", optim_strategy)

    return train_gen, train_disc, gen_loss, disc_loss, gp_loss, gen_sample, g_gradients, g_variables, d_gradients, d_variables, max_g_norm, max_d_norm

    # Op to update the learning rate according to a schedule
def lr_update(lr, intra_phase_step, steps_per_phase, lr_max, lr_increase, lr_decrease, lr_rise_niter, lr_decay_niter):
    """Op that updates the learning rate according to a schedule.
    Args:
      lr: Tensor which contains the current learning rate that needs to be updated
      intra_phase_step: Step counter representing the number of images processed since the start of the current phase
      steps_per_phase: total number of steps in a phase
      lr_max: learning rate after increase (and before decrease) segments
      lr_increase: type of increase function to use (e.g. None, linear, exponential)
      lr_decrease: type of decrease function to use (e.g. None, linear, exponential)
      lr_rise_niter: number of iterations over which the increase from the minimum to the maximum value should happen
      lr_decay_niter: number of iterations over which the decrease from the maximum to the minumum value should happen.
    Returns: an Op that can be passed to session.run to update the learning (lr) Tensor
    """

    # Default starting point is that update_lr = lr_max. If there are no lr_increase or lr_decrease
    # functions specified, it stays like this.
    lr_update = lr_max

    # Is a learning rate schedule defined at all? (otherwise, immediately return a constant)
    if (lr_increase or lr_decrease):
        # Rather than if-else statements, the way to define a piecewiese function is through tf.cond

        # Prepare some variables:
        a = tf.cast(tf.math.divide(lr_max, 100), tf.float32)
        b_rise = tf.cast(tf.math.divide(np.log(100), lr_rise_niter), tf.float32)
        b_decay = tf.cast(tf.math.divide(np.log(100), lr_decay_niter), tf.float32)
        step_decay_start = tf.subtract(steps_per_phase, lr_decay_niter)
        remaining_steps = tf.subtract(steps_per_phase, intra_phase_step)

        # Define the different functions
        def update_increase_lin ():
            return tf.multiply(
                               tf.cast(tf.truediv(intra_phase_step, lr_rise_niter), tf.float32),
                               lr_max
                               )
        def update_increase_exp():
            return tf.multiply(
                                a,
                                tf.math.exp(tf.multiply(b_rise, tf.cast(intra_phase_step, tf.float32)))
                                )

        def update_decrease_lin():
            return tf.multiply(
                               tf.cast(tf.truediv(remaining_steps, lr_decay_niter), tf.float32),
                               lr_max
                               )

        def update_decrease_exp():
            return tf.multiply(
                                a,
                                tf.math.exp(tf.multiply(b_decay, tf.cast(remaining_steps, tf.float32)))
                                )

        def no_op():
            return lr_update

        if lr_increase == 'linear':
            # Are we in the increasing part? Return update_increase_lin function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step < lr_rise_niter, update_increase_lin, no_op)
        elif lr_increase == 'exponential':
            # Are we in the increasing part? Return update_increase_exp function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step < lr_rise_niter, update_increase_exp, no_op)
            
        if lr_decrease == 'linear':
            # Are we in the decreasing part? Return return update_decrease function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step > step_decay_start, update_decrease_lin, no_op) 
        elif lr_decrease == 'exponential':
            # Are we in the decreasing part? Return return update_decrease function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step > step_decay_start, update_decrease_exp, no_op) 
 
    return lr.assign(lr_update)