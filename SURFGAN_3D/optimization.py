import tensorflow as tf
import numpy as np

from networks.loss import forward_simultaneous, forward_generator, forward_discriminator

def get_optimizer(d_lr, g_lr, args):
    """Check args.optimizer and return a tf.Optimizer object. The tf.Optimizer is initialized with parameters from args.
    Parameters:
        d_lr: A tf.Variable representing the discriminator learning rate. We use a tf.Variable, and not args.d_lr directly since this allows adaptation of the learning rate during training.
        g_lr: A tf.Variable representing the generator learning rate.
        args: the parsed command line arguments.
    Returns: optimizer_gen, optimizer_disc, a tuple with optimizers for the generator and discriminator, respectively (tf.Optimizer, tf.Optimizer)"""

    # Create the right optimizer
    if args.optimizer == 'Adam':
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=args.adam_beta1, beta2=args.adam_beta2)
    elif args.optimizer == 'SGD':
        optimizer_gen = tf.train.GradientDescentOptimizer(learning_rate=g_lr)
    elif args.optimizer == 'Adadelta':
        optimizer_gen = tf.train.AdadeltaOptimizer(learning_rate=g_lr, rho=args.rho, epsilon=1e-07)
    elif args.optimizer == 'Momentum':
        optimizer_gen = tf.train.MomentumOptimizer(learning_rate = g_lr, momentum = args.momentum, use_nesterov = True)
    else:
        print(f"ERROR: optimizer argument {args.optimizer} not recognized or implemented")
        raise NotImplementedError

    if args.d_optimizer == 'Adam':
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=args.d_adam_beta1, beta2=args.d_adam_beta2)
    elif args.d_optimizer == 'SGD':
        optimizer_disc = tf.train.GradientDescentOptimizer(learning_rate=d_lr)
    elif args.d_optimizer == 'Adadelta':
        optimizer_disc = tf.train.AdadeltaOptimizer(learning_rate=d_lr, rho=args.d_rho, epsilon=1e-07)
    elif args.d_optimizer == 'Momentum':
        optimizer_disc = tf.train.MomentumOptimizer(learning_rate = d_lr, momentum = args.d_momentum, use_nesterov = True)
    else:
        print(f"ERROR: optimizer argument {args.d_optimizer} not recognized or implemented")
        raise NotImplementedError

    #optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=g_lr)
    #optimizer_disc = tf.train.RMSPropOptimizer(learning_rate=d_lr)
    
    # optimizer_gen = RAdamOptimizer(learning_rate=g_lr, beta1=args.beta1, beta2=args.beta2)
    # optimizer_disc = RAdamOptimizer(learning_rate=d_lr, beta1=args.beta1, beta2=args.beta2)

    return optimizer_gen, optimizer_disc

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
    num_phases, base_dim, base_shape, kernel_shape, kernel_spec, filter_spec, activation, leakiness, network_size, loss_fn, gp_weight, optim_strategy, g_clipping, d_clipping, noise_stddev, freeze_vars=None):
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
        kernel_shape: desired kernel shape (the spatial part, e.g. for 3D images, [3,3,3] would be a valid kernel_shape)
        activation:
        leakiness:
        network_size:
        loss_fn:
        gp_weight:
        optim_strategy:
        g_clipping:
        d_clipping:
        noise_stddev:
        freeze_vars: list of variables that will not be updated when the train_gen_freeze or train_disc_freese ops are called. Typically, this is the list of variables from previous phases
    Returns:
        train_gen, train_disc, gen_loss, disc_loss, gp_loss, gen_sample, g_gradients, g_variables, d_gradients, d_variables, max_g_norm, max_d_norm, train_gen_freeze, g_gradients_freeze, g_variables_freeze, max_g_norm_freeze, train_disc_freeze, d_gradients_freeze, d_variables_freeze, max_d_norm_freeze
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
        max_g_norm:
        max_d_norm:
        train_gen_freeze: generator training op where freeze_vars are not updated (returns None if freeze_vars is undefined)
        g_gradients_freeze: generator gradients for non-frozen variables (returns None if freeze_vars is undefined)
        g_variables_freeze: generator variables
        max_g_norm_freeze:
        train_disc_freeze:
        d_gradients_freeze:
        d_variables_freeze:
        max_d_norm_freeze:
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
            kernel_shape,
            kernel_spec, filter_spec,
            activation,
            leakiness,
            network_size,
            loss_fn,
            gp_weight,
            noise_stddev
        )

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        train_gen, g_gradients, g_variables, max_g_norm = minimize_with_clipping(optimizer_gen, gen_loss, gen_vars, g_clipping)

        gen_vars_limited = None; train_gen_feeze = None; g_gradients_freeze = None; g_variables_freeze = None; max_g_norm_freeze = None
        if freeze_vars is not None:
            gen_vars_limited = [var for var in gen_vars if var.name not in [x.name for x in freeze_vars]]
            print(f'gen_vars_limited: {gen_vars_limited}')
            train_gen_freeze, g_gradients_freeze, g_variables_freeze, max_g_norm_freeze = minimize_with_clipping(optimizer_gen, gen_loss, gen_vars_limited, g_clipping)
        

        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        train_disc, d_gradients, d_variables, max_d_norm = minimize_with_clipping(optimizer_disc, disc_loss, disc_vars, d_clipping)

        disc_vars_limited = None; train_disc_freeze = None; d_gradients_freeze = None; d_variables_freeze = None; max_d_norm_freeze = None
        if freeze_vars is not None:
            disc_vars_limited = [var for var in disc_vars if var.name not in [x.name for x in freeze_vars]]
            print(f'disc_vars_limited: {disc_vars_limited}')
            train_disc_freeze, d_gradients_freeze, d_variables_freeze, max_d_norm_freeze = minimize_with_clipping(optimizer_disc, disc_loss, disc_vars_limited, d_clipping)
        

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
            kernel_shape,
            kernel_spec, filter_spec,
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
        
        if freeze_vars is not None:
            disc_vars_limited = [var for var in disc_vars if var.name not in [x.name for x in freeze_vars]]
            print('disc_vars_limited: {disc_vars_limited}')
            train_disc_freeze, d_gradients_freeze, d_variables_freeze, max_d_norm_freeze = minimize_with_clipping(optimizer_disc, disc_loss, disc_vars_limited, d_clipping)

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
                kernel_shape,
                kernel_spec, filter_spec,
                activation,
                leakiness,
                network_size,
                loss_fn,
                noise_stddev,
                is_reuse=True
            )

            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            train_gen, g_gradients, g_variables, max_g_norm = minimize_with_clipping(optimizer_gen, gen_loss, gen_vars, g_clipping)

            if freeze_vars is not None:
                gen_vars_limited = [var for var in gen_vars if var.name not in [x.name for x in freeze_vars]]
                print('gen_vars_limited: {gen_vars_limited}')
                train_gen_freeze, g_gradients_freeze, g_variables_freeze, max_g_norm_freeze = minimize_with_clipping(optimizer_gen, gen_loss, gen_vars_limited, g_clipping)

    else:
        raise ValueError("Unknown optim strategy ", optim_strategy)

    return train_gen, train_disc, gen_loss, disc_loss, gp_loss, gen_sample, g_gradients, g_variables, \
        d_gradients, d_variables, max_g_norm, max_d_norm, \
        train_gen_freeze, g_gradients_freeze, g_variables_freeze, max_g_norm_freeze, \
        train_disc_freeze, d_gradients_freeze, d_variables_freeze, max_d_norm_freeze

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