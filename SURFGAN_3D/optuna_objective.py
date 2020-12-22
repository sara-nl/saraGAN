# # pylint: disable=import-error
# import argparse
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import time
import random
# import optuna
# # from signal import signal, SIGSEGV

from utils import count_parameters, image_grid, parse_tuple, MPMap, log0, lr_update
from utils import get_compute_metrics_dict, get_logdir, get_verbosity, get_filewriter, get_base_shape, get_num_phases, get_num_channels
from utils import get_num_metric_samples, scale_lr, get_xy_dim, get_numpy_dataset, get_current_input_shape
from optuna_suggestions import optuna_override_undefined
# from mpi4py import MPI
import os
import importlib
# from rectified_adam import RAdamOptimizer
from networks.loss import forward_simultaneous, forward_generator, forward_discriminator
# import psutil
# from networks.ops import num_filters
# from tensorflow.data.experimental import AUTOTUNE
# import nvgpu
# import logging

from metrics.save_metrics import save_metrics



def optuna_objective(trial, args, config):

    # Store the last fid so that it can be returned to optuna
    last_fid = None

    # Override args.* that are undefined by optuna's suggest_* calls.
    # For now, this is limited to overriding learning rate, batch size, and learning rate schedules, but may be expanded in the future (see optuna_suggestions.py)
    # Note: this means that when restoring from an optuna FrozenTrial, command line parameters take precedence!
    args = optuna_override_undefined(args, trial)

    # Importing modules by name for the generator and discriminator
    discriminator = importlib.import_module(f'networks.{args.architecture}.discriminator').discriminator
    generator = importlib.import_module(f'networks.{args.architecture}.generator').generator

        # Set verbosity:
    verbose = get_verbosity(args.horovod, args.optuna_distributed)
    if not verbose:
        tf.get_logger().setLevel(logging.ERROR) # Only errors if rank != 0

    # set world size
    if args.horovod:
        global_size = hvd.size()
    else:
        global_size = 1

    # Get logging directory based on the args. If args.logdir is not set, a logdir is created
    logdir = get_logdir(args)
    # Returns a tf.FileWriter, but only for rank 0 if the training uses multiple MPI ranks
    writer = get_filewriter(logdir, verbose)

    # Allow GPU memory growth
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Get starting & final resolutions
    start_shape = parse_tuple(args.start_shape)
    start_resolution = start_shape[-1]
    final_shape = parse_tuple(args.final_shape)
    image_channels = final_shape[0]
    final_resolution = final_shape[-1]

    # Number of phases required to get from the starting resolution to the final resolution
    num_phases = int(np.log2(final_resolution/start_resolution))

    # Define the shape at the base of the network
    base_shape = (image_channels, start_shape[1], start_shape[2], start_shape[3])

    # Number of filters at the base (= 1st convolutional layer of the generator) of the progressive network
    # In subsequent phases, the number of filters will go down as the resolution goes up.
    base_dim = args.first_conv_nfilters

    if verbose:
        print(f"Deduced number of phases: {num_phases}")
        print(f"base_dim: {base_dim}")

    var_list = list()
    global_step = 0

    for phase in range(1, num_phases + 1):

        tf.reset_default_graph()
        # Random seeds need to be reinitialized after a reset_default_graph (at least for TF, but I guess resetting all is good)
        if args.horovod:
            np.random.seed(args.seed + hvd.rank())
            tf.random.set_random_seed(args.seed + hvd.rank())
            random.seed(args.seed + hvd.rank())
        else:
            np.random.seed(args.seed)
            tf.random.set_random_seed(args.seed)
            random.seed(args.seed)

        # ------------------------------------------------------------------------------------------#
        # DATASET

        size = start_resolution * (2 ** (phase - 1))

        # Get NumpyPathDataset object for current phase. It's an iterable object that returns the path to samples in the dataset
        npy_data = get_numpy_dataset(phase, args.starting_phase, args.start_shape, args.dataset_path, args.scratch_path, verbose)

        # # dataset = tf.data.Dataset.from_generator(npy_data.__iter__, npy_data.dtype, npy_data.shape)
        # dataset = tf.data.Dataset.from_tensor_slices(npy_data.scratch_files)

        # Use optuna to explore the base_batch_size. We sample the exponent, so that we sample from (1, 2, 4, 8, ..., 1024)
        # args.base_batch_size = 2 ** trial.suggest_int('base_batch_size_exponent', 1, 6)

        # Get DataLoader
        batch_size = max(1, args.base_batch_size // (2 ** (phase - 1)))

        if phase >= args.starting_phase:
            # assert batch_size * global_size <= args.max_global_batch_size
            if verbose:
                print(f"Using local batch size of {batch_size} and global batch size of {batch_size * global_size}")

        # Num_metric_samples is the amount of samples the metric is calculated on.
        # If it is not set explicitely, we use the same as the global batch size, but never less than 2 per worker (1 per worker potentially makes some metrics crash)
        if not args.num_metric_samples:
            if batch_size > 1:
                num_metric_samples = batch_size * global_size
            else:
                num_metric_samples = 2 * global_size
        else:
            num_metric_samples = args.num_metric_samples

        # if args.horovod:
        #     dataset.shard(hvd.size(), hvd.rank())
        #
        # def load(x):
        #     x = np.load(x.decode())[np.newaxis, ...].astype(np.float32) / 1024 - 1
        #     return x
        #
        # if args.gpu:
        #     parallel_calls = AUTOTUNE
        # else:
        #     parallel_calls = int(os.environ['OMP_NUM_THREADS'])
        #
        # dataset = dataset.shuffle(len(npy_data))
        # dataset = dataset.map(lambda x: tuple(tf.py_func(load, [x], [tf.float32])), num_parallel_calls=parallel_calls)
        # dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.repeat()
        # dataset = dataset.prefetch(AUTOTUNE)
        # dataset = dataset.make_one_shot_iterator()
        # data = dataset.get_next()
        # if len(data) == 1:
        #     real_image_input = data
        #     real_label = None
        # elif len(data) == 2:
        #     real_image_input, real_label = data
        # else:
        #     raise NotImplementedError()

#zdim_base = max(1, final_shape[1] // (2 ** num_phases))
        current_shape = [batch_size, image_channels, *[size * 2 ** (phase - 1) for size in
                                                       base_shape[1:]]]
        if verbose:
            print(f'base_shape: {base_shape}, current_shape: {current_shape}')
        real_image_input = tf.placeholder(shape=current_shape, dtype=tf.float32)

        # real_image_input = tf.random.normal([1, batch_size, image_channels, *[size * 2 ** (phase -
        #                                                                                  1) for size in base_shape[1:]]])
        # real_image_input = tf.squeeze(real_image_input, axis=0)
        # real_image_input = tf.ensure_shape(real_image_input, [batch_size, image_channels, *[size * 2 ** (phase - 1) for size in base_shape[1:]]])
        real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * .01
        real_label = None

        if real_label is not None:
            real_label = tf.one_hot(real_label, depth=args.num_labels)

        # ------------------------------------------------------------------------------------------#
        # OPTIMIZERS

        # # Use optuna for LR, overwrite original args
        # args.g_lr = trial.suggest_loguniform('generator_LR', 1e-6, 1e-2)
        # args.d_lr = trial.suggest_loguniform('discriminator_LR', 1e-6, 1e-2)

        # # Use optuna for LR schedules. Predefine some schedules, as not all make sense
        # lr_schedule = [
        #     {'lr_sched': None, 'lr_fract': 0.5},
        #     {'lr_sched': 'linear', 'lr_fract': 0.125},
        #     {'lr_sched': 'linear', 'lr_fract': 0.25},
        #     {'lr_sched': 'linear', 'lr_fract': 0.375},
        #     {'lr_sched': 'linear', 'lr_fract': 0.5},
        #     {'lr_sched': 'exponential', 'lr_fract': 0.125},
        #     {'lr_sched': 'exponential', 'lr_fract': 0.25},
        #     {'lr_sched': 'exponential', 'lr_fract': 0.375},
        #     {'lr_sched': 'exponential', 'lr_fract': 0.5},
        # ]

        # # d_lr_sched_inc = trial.suggest_categorical('d_lr_sched_inc', lr_schedule)
        # # d_lr_sched_dec = trial.suggest_categorical('d_lr_sched_dec', lr_schedule)
        # # g_lr_sched_inc = trial.suggest_categorical('g_lr_sched_inc', lr_schedule)
        # # g_lr_sched_dec = trial.suggest_categorical('g_lr_sched_dec', lr_schedule)

        # # args.g_lr_increase = lr_schedule[g_lr_sched_inc]['lr_sched']
        # # args.g_lr_decrease = lr_schedule[g_lr_sched_dec]['lr_sched']
        # # args.d_lr_increase = lr_schedule[d_lr_sched_inc]['lr_sched']
        # # args.d_lr_decrease = lr_schedule[d_lr_sched_dec]['lr_sched']

        # # args.g_lr_rise_niter = np.ceil(lr_schedule[g_lr_sched_inc]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg))
        # # args.g_lr_dec_niter = np.ceil(lr_schedule[g_lr_sched_dec]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg))
        # # args.d_lr_rise_niter = np.ceil(lr_schedule[d_lr_sched_inc]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg))
        # # args.d_lr_dec_niter = np.ceil(lr_schedule[d_lr_sched_dec]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg))

        # d_lr_sched_inc = trial.suggest_categorical('d_lr_sched_inc', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        # d_lr_sched_dec = trial.suggest_categorical('d_lr_sched_dec', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        # g_lr_sched_inc = trial.suggest_categorical('g_lr_sched_inc', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        # g_lr_sched_dec = trial.suggest_categorical('g_lr_sched_dec', [0, 1, 2, 3, 4, 5, 6, 7, 8])

        # args.g_lr_increase = lr_schedule[g_lr_sched_inc]['lr_sched']
        # args.g_lr_decrease = lr_schedule[g_lr_sched_dec]['lr_sched']
        # args.d_lr_increase = lr_schedule[d_lr_sched_inc]['lr_sched']
        # args.d_lr_decrease = lr_schedule[d_lr_sched_dec]['lr_sched']

        # args.g_lr_rise_niter = np.ceil(lr_schedule[g_lr_sched_inc]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
        # args.g_lr_dec_niter = np.ceil(lr_schedule[g_lr_sched_dec]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
        # args.d_lr_rise_niter = np.ceil(lr_schedule[d_lr_sched_inc]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
        # args.d_lr_dec_niter = np.ceil(lr_schedule[d_lr_sched_dec]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)


        g_lr = args.g_lr
        d_lr = args.d_lr
        

        if args.horovod:
            if args.g_scaling == 'sqrt':
                g_lr = g_lr * np.sqrt(hvd.size())
            elif args.g_scaling == 'linear':
                g_lr = g_lr * hvd.size()
            elif args.g_scaling == 'none':
                pass
            else:
                raise ValueError(args.g_scaling)

            if args.d_scaling == 'sqrt':
                d_lr = d_lr * np.sqrt(hvd.size())
            elif args.d_scaling == 'linear':
                d_lr = d_lr * hvd.size()
            elif args.d_scaling == 'none':
                pass
            else:
                raise ValueError(args.d_scaling)

        d_lr = tf.Variable(d_lr, name='d_lr', dtype=tf.float32)
        g_lr = tf.Variable(g_lr, name='g_lr', dtype=tf.float32)

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=args.beta1, beta2=args.beta2)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=args.beta1, beta2=args.beta2)
        #optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=g_lr)
        #optimizer_disc = tf.train.RMSPropOptimizer(learning_rate=d_lr)
        # optimizer_gen = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        # optimizer_disc = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        # optimizer_gen = RAdamOptimizer(learning_rate=g_lr, beta1=args.beta1, beta2=args.beta2)
        # optimizer_disc = RAdamOptimizer(learning_rate=d_lr, beta1=args.beta1, beta2=args.beta2)

        intra_phase_step = tf.Variable(0, name='step', dtype=tf.int32)
        update_intra_phase_step = intra_phase_step.assign_add(batch_size*global_size)

        # Turn arguments into constant Tensors
        g_lr_max = tf.constant(args.d_lr, tf.float32)
        d_lr_max = tf.constant(args.g_lr, tf.float32)
#        g_lr_rise_niter = tf.constant(args.g_lr_rise_niter)
#        d_lr_rise_niter = tf.constant(args.d_lr_rise_niter)
#        g_lr_decay_niter = tf.constant(args.g_lr_decay_niter)
#        d_lr_decay_niter = tf.constant(args.d_lr_decay_niter)
        steps_per_phase = tf.constant(args.mixing_nimg + args.stabilizing_nimg)

#        with tf.control_dependencies([update_intra_phase_step]):
#            update_g_lr = g_lr.assign(g_lr * args.g_annealing)
#            update_d_lr = d_lr.assign(d_lr * args.d_annealing)
        update_g_lr = lr_update(lr = g_lr, intra_phase_step = intra_phase_step, 
                                     steps_per_phase = steps_per_phase, lr_max = g_lr_max,
                                     lr_increase = args.g_lr_increase, lr_decrease = args.g_lr_decrease,
                                     lr_rise_niter = args.g_lr_rise_niter, lr_decay_niter = args.g_lr_decay_niter
                                    )
        update_d_lr = lr_update(lr = d_lr, intra_phase_step = intra_phase_step, 
                                     steps_per_phase = steps_per_phase, lr_max = d_lr_max,
                                     lr_increase = args.d_lr_increase, lr_decrease = args.d_lr_decrease,
                                     lr_rise_niter = args.d_lr_rise_niter, lr_decay_niter = args.d_lr_decay_niter
                                    )

        if args.horovod:
            if args.use_adasum:
                # optimizer_gen = hvd.DistributedOptimizer(optimizer_gen, op=hvd.Adasum)
                optimizer_gen = hvd.DistributedOptimizer(optimizer_gen)
                optimizer_disc = hvd.DistributedOptimizer(optimizer_disc, op=hvd.Adasum)
            else:
                optimizer_gen = hvd.DistributedOptimizer(optimizer_gen)
                optimizer_disc = hvd.DistributedOptimizer(optimizer_disc)

        # ------------------------------------------------------------------------------------------#
        # NETWORKS

        with tf.variable_scope('alpha'):
            alpha = tf.Variable(1, name='alpha', dtype=tf.float32)
            # Alpha init
            init_alpha = alpha.assign(1)

            # Specify alpha update op for mixing phase.
            num_steps = args.mixing_nimg // (batch_size * global_size)
            # This original code produces too large steps when performing a run that is restarted in the middle of the alpha mixing phase:
            # alpha_update = 1 / num_steps
            # This code produces a correct step size when restarting (the same step size that would be used if a run wasn't restarted)
            alpha_update = args.starting_alpha / num_steps
            # noinspection PyTypeChecker
            update_alpha = alpha.assign(tf.maximum(alpha - alpha_update, 0))

        if args.optim_strategy == 'simultaneous':
            gen_loss, disc_loss, gp_loss, gen_sample = forward_simultaneous(
                generator,
                discriminator,
                real_image_input,
                args.latent_dim,
                alpha,
                phase,
                num_phases,
                base_dim,
                base_shape,
                args.activation,
                args.leakiness,
                args.network_size,
                args.loss_fn,
                args.gp_weight
            )
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            g_gradients, g_variables = zip(*optimizer_gen.compute_gradients(gen_loss,
                                                                            var_list=gen_vars))
            if args.g_clipping:
                g_gradients, _ = tf.clip_by_global_norm(g_gradients, 1.0)


            d_gradients, d_variables = zip(*optimizer_disc.compute_gradients(disc_loss,
                                                                             var_list=disc_vars))
            if args.d_clipping:
                d_gradients, _ = tf.clip_by_global_norm(d_gradients, 1.0)


            g_norms = tf.stack([tf.norm(grad) for grad in g_gradients if grad is not None])
            max_g_norm = tf.reduce_max(g_norms)
            d_norms = tf.stack([tf.norm(grad) for grad in d_gradients if grad is not None])
            max_d_norm = tf.reduce_max(d_norms)

            # g_norms = tf.stack([tf.norm(grad) for grad, var in g_gradients if grad is not None])
            # max_g_norm = tf.reduce_max(g_norms)
            # d_norms = tf.stack([tf.norm(grad) for grad, var in d_gradients if grad is not None])
            # max_d_norm = tf.reduce_max(d_norms)

            # g_clipped_grads = [(tf.clip_by_norm(grad, clip_norm=128), var) for grad, var in g_gradients]
            # train_gen = optimizer_gen.apply_gradients(g_clipped_grads)
            train_gen = optimizer_gen.apply_gradients(zip(g_gradients, g_variables))
            train_disc = optimizer_disc.apply_gradients(zip(d_gradients, d_variables))

            # train_gen = optimizer_gen.apply_gradients(g_gradients)
            # train_disc = optimizer_disc.apply_gradients(d_gradients)

        elif args.optim_strategy == 'alternate':

            disc_loss, gp_loss = forward_discriminator(
                generator,
                discriminator,
                real_image_input,
                args.latent_dim,
                alpha,
                phase,
                num_phases,
                args.base_dim,
                base_shape,
                args.activation,
                args.leakiness,
                args.network_size,
                args.loss_fn,
                args.gp_weight,
                # conditioning=real_label
            )

            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            d_gradients = optimizer_disc.compute_gradients(disc_loss, var_list=disc_vars)
            d_norms = tf.stack([tf.norm(grad) for grad, var in d_gradients if grad is not None])
            max_d_norm = tf.reduce_max(d_norms)

            train_disc = optimizer_disc.apply_gradients(d_gradients)

            with tf.control_dependencies([train_disc]):
                gen_sample, gen_loss = forward_generator(
                    generator,
                    discriminator,
                    real_image_input,
                    args.latent_dim,
                    alpha,
                    phase,
                    num_phases,
                    base_dim,
                    base_shape,
                    args.activation,
                    args.leakiness,
                    args.network_size,
                    args.loss_fn,
                    is_reuse=True
                )

                gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                g_gradients = optimizer_gen.compute_gradients(gen_loss, var_list=gen_vars)
                g_norms = tf.stack([tf.norm(grad) for grad, var in g_gradients if grad is not None])
                max_g_norm = tf.reduce_max(g_norms)
                train_gen = optimizer_gen.apply_gradients(g_gradients)

        else:
            raise ValueError("Unknown optim strategy ", args.optim_strategy)

        if verbose:
            print(f"Generator parameters: {count_parameters('generator')}")
            print(f"Discriminator parameters:: {count_parameters('discriminator')}")

        # train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
        # train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

        ema = tf.train.ExponentialMovingAverage(decay=args.ema_beta)
        ema_op = ema.apply(gen_vars)
        # Transfer EMA values to original variables
        ema_update_weights = tf.group(
            [tf.assign(var, ema.average(var)) for var in gen_vars])

        with tf.name_scope('summaries'):
            # We want to store large / heavy summaries like images less frequently
            summary_small = []
            summary_large = []
            # Summaries
            summary_small.append(tf.summary.scalar('d_loss', disc_loss))
            summary_small.append(tf.summary.scalar('g_loss', gen_loss))
            summary_small.append(tf.summary.scalar('gp', tf.reduce_mean(gp_loss)))

            for g in zip(g_gradients, g_variables):
                summary_small.append(tf.summary.histogram(f'grad_{g[1].name}', g[0]))

            for g in zip(d_gradients, d_variables):
                summary_small.append(tf.summary.histogram(f'grad_{g[1].name}', g[0]))

            # tf.summary.scalar('convergence', tf.reduce_mean(disc_real) - tf.reduce_mean(tf.reduce_mean(disc_fake_d)))

            summary_small.append(tf.summary.scalar('max_g_grad_norm', max_g_norm))
            summary_small.append(tf.summary.scalar('max_d_grad_norm', max_d_norm))

            # Spread out 3D image as 2D grid, slicing in the z-dimension
            real_image_grid = tf.transpose(real_image_input[0], (1, 2, 3, 0))
            shape = real_image_grid.get_shape().as_list()
            print(f'real_image_grid shape: {shape}')
            grid_cols = int(2 ** np.floor(np.log(np.sqrt(shape[0])) / np.log(2)))
            # If the image z-dimension isn't divisible by grid_rows, we need to pad
            if (shape[0] % grid_cols) != 0:
                # Initialize pad_list for numpy padding
                pad_list = [[0,0] for i in range(0, len(shape))]
                # Compute number of slices we need to add to get to the next multiple of shape[0]
                pad_nslices = grid_cols - (shape[0] % grid_cols)
                pad_list[0] = [0, pad_nslices]
                real_image_grid = tf.pad(real_image_grid, tf.constant(pad_list), "CONSTANT", constant_values=0)
                # Recompute shape, so that the number of grid_rows is adapted to that
                shape = real_image_grid.get_shape().as_list()
            grid_rows = int(np.ceil(shape[0] / grid_cols))
            grid_shape = [grid_rows, grid_cols]
            real_image_grid = image_grid(real_image_grid, grid_shape, image_shape=shape[1:3],
                                         num_channels=shape[-1])

            fake_image_grid = tf.transpose(gen_sample[0], (1, 2, 3, 0))
            # Use the same padding for the fake_image_grid
            if (fake_image_grid.get_shape().as_list()[0] % grid_cols) != 0:
                fake_image_grid = tf.pad(fake_image_grid, tf.constant(pad_list), "CONSTANT", constant_values=0)
            fake_image_grid = image_grid(fake_image_grid, grid_shape, image_shape=shape[1:3],
                                         num_channels=shape[-1])

            fake_image_grid = tf.clip_by_value(fake_image_grid, -1, 2)

            summary_large.append(tf.summary.image('real_image', real_image_grid))
            summary_large.append(tf.summary.image('fake_image', fake_image_grid))

            summary_small.append(tf.summary.scalar('fake_image_min', tf.math.reduce_min(gen_sample)))
            summary_small.append(tf.summary.scalar('fake_image_max', tf.math.reduce_max(gen_sample)))

            summary_small.append(tf.summary.scalar('real_image_min', tf.math.reduce_min(real_image_input[0])))
            summary_small.append(tf.summary.scalar('real_image_max', tf.math.reduce_max(real_image_input[0])))
            summary_small.append(tf.summary.scalar('alpha', alpha))

            summary_small.append(tf.summary.scalar('g_lr', g_lr))
            summary_small.append(tf.summary.scalar('d_lr', d_lr))

            # merged_summaries = tf.summary.merge_all()
            summary_small = tf.summary.merge(summary_small)
            summary_large = tf.summary.merge(summary_large)

        # Other ops
        init_op = tf.global_variables_initializer()
        assign_starting_alpha = alpha.assign(args.starting_alpha)
        assign_zero = alpha.assign(0)
        broadcast = hvd.broadcast_global_variables(0)
        #print("Global variables:")
        #print("%s" % tf.compat.v1.global_variables())

        with tf.Session(config=config) as sess:
            # if args.gpu:
            #     assert tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
            # sess.graph.finalize()
            sess.run(init_op)

            trainable_variable_names = [v.name for v in tf.trainable_variables()]

            if var_list is not None and phase > args.starting_phase:
                print("Restoring variables from:", os.path.join(logdir, f'model_{phase - 1}'))
                var_names = [v.name for v in var_list]
                load_vars = [sess.graph.get_tensor_by_name(n) for n in var_names if n in trainable_variable_names]
                saver = tf.train.Saver(load_vars)
                saver.restore(sess, os.path.join(logdir, f'model_{phase - 1}'))
                print("Variables restored!")
            elif var_list is not None and args.continue_path and phase == args.starting_phase:
                print("Restoring variables from:", args.continue_path)
                var_names = [v.name for v in var_list]
                load_vars = [sess.graph.get_tensor_by_name(n) for n in var_names if n in trainable_variable_names]
                saver = tf.train.Saver(load_vars)
                saver.restore(sess, os.path.join(args.continue_path))
                print("Variables restored!")
            else:
                if verbose:
                     print("Not restoring variables.")
                     print("Variable List Length:", len(var_list))
                     writer.add_graph(sess.graph)

            var_list = gen_vars + disc_vars

            if phase < args.starting_phase:
                continue

            if phase == args.starting_phase:
                sess.run(assign_starting_alpha)
            else:
                sess.run(init_alpha)

            if verbose:
                print(f"Begin mixing epochs in phase {phase}")
            if args.horovod:
                if verbose:
                    print("Broadcasting initial global variables...")
                sess.run(broadcast)
                if verbose:
                    print("Broadcast completed")

            local_step = 0
            # take_first_snapshot = True

            if args.optuna_distributed:
                print(f"Trial: {trial.number}, Worker: {hvd.rank()}, Parameters: {trial.params}")
            else:
                print(f"Trial: {trial.number}, Parameters: {trial.params}")

            while True:
                start = time.time()

                # Update learning rate
                d_lr_val = sess.run(update_d_lr)
                g_lr_val = sess.run(update_g_lr)

                if global_step % args.checkpoint_every_nsteps < (batch_size*global_size) and local_step > 0:
                    if args.horovod:
                        if verbose:
                            print("Broadcasting global variables for checkpointing...")
                        sess.run(broadcast)
                        if verbose:
                            print("Broadcast completed")
                    saver = tf.train.Saver(var_list)
                    if verbose:
                        print(f'Writing checkpoint file: model_{phase}_ckpt_{global_step}')
                        saver.save(sess, os.path.join(logdir, f'model_{phase}_ckpt_{global_step}'))

                #print("Batching...")
                batch_loc = np.random.randint(0, len(npy_data) - batch_size)
                batch_paths = npy_data[batch_loc: batch_loc + batch_size]
                batch = np.stack([np.load(path) for path in batch_paths])
                batch = batch[:, np.newaxis, ...].astype(np.float32) / 1024 - 1
                #print("Got a batch!")

                #sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
                #sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6789')
                small_summary_bool = (local_step % args.summary_small_every_nsteps == 0)
                large_summary_bool = (local_step % args.summary_large_every_nsteps == 0)
                metrics_summary_bool = (local_step % args.metrics_every_nsteps == 0)
                if large_summary_bool:
                    _, _, summary_s, summary_l, d_loss, g_loss = sess.run(
                         [train_gen, train_disc, summary_small, summary_large,
                          disc_loss, gen_loss], feed_dict={real_image_input: batch})
                elif small_summary_bool:
                    _, _, summary_s, d_loss, g_loss = sess.run(
                         [train_gen, train_disc, summary_small,
                          disc_loss, gen_loss], feed_dict={real_image_input: batch})
                else:
                    _, _, d_loss, g_loss = sess.run(
                         [train_gen, train_disc, disc_loss, gen_loss],
                         feed_dict={real_image_input: batch})
                #print("Completed step")
                global_step += batch_size * global_size
                local_step += 1

                end = time.time()
                local_img_s = batch_size / (end - start)
                img_s = global_size * local_img_s

                sess.run(update_alpha)
                sess.run(ema_op)
                in_phase_step = sess.run(update_intra_phase_step)

                if metrics_summary_bool:
                    if args.calc_metrics:
                        # if verbose:
                            # print('Computing and writing metrics...')
                        metrics = save_metrics(writer, sess, npy_data, gen_sample, batch_size, global_size, global_step, size, args.horovod, get_compute_metrics_dict(args), num_metric_samples, verbose)

                        # Optuna pruning and return value:
                        last_fid = metrics['FID']
                        if args.optuna_distributed:
                            print(f"Trial: {trial.number}, Worker: {hvd.rank()}, Parameters: {trial.params}, global_step: {global_step}, FID: {last_fid}")
                        else:
                            print(f"Trial: {trial.number}, Parameters: {trial.params}, global_step: {global_step}, FID: {last_fid}")
                        trial.report(metrics['FID'], global_step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                if verbose:
                    if large_summary_bool:
                        print('Writing large summary...')
                        writer.add_summary(summary_s, global_step)
                        writer.add_summary(summary_l, global_step)
                    elif small_summary_bool:
                        print('Writing small summary...')
                        writer.add_summary(summary_s, global_step)
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='img_s', simple_value=img_s)]),
                                           global_step)
                    # memory_percentage = psutil.Process(os.getpid()).memory_percent()
                    # if not args.gpu:
                    #     memory_percentage = psutil.Process(os.getpid()).memory_percent()
                    # else:
                    #     memory_percentage = nvgpu.gpu_info()[local_rank]['mem_used_percent']


                    # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='memory_percentage', simple_value=memory_percentage)]),
                    #                    global_step)
                    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
                    # print(f"{current_time} \t"
                    #       f"Step {global_step:09} \t"
                    #       f"Step(phase) {in_phase_step:09} \t"
                    #       f"img/s {img_s:.2f} \t "
                    #       f"img/s/worker {local_img_s:.3f} \t"
                    #       f"d_loss {d_loss:.4f} \t "
                    #       f"g_loss {g_loss:.4f} \t "
                    #       f"d_lr {d_lr_val:.5f} \t"
                    #       f"g_lr {g_lr_val:.5f} \t"
                    #       # f"memory {memory_percentage:.4f} % \t"
                    #       f"alpha {alpha.eval():.2f}")

                #     # if take_first_snapshot:
                #     #     import tracemalloc
                #     #     tracemalloc.start()
                #     #     snapshot_first = tracemalloc.take_snapshot()
                #     #     take_first_snapshot = False

                #     # snapshot = tracemalloc.take_snapshot()
                #     # top_stats = snapshot.compare_to(snapshot_first, 'lineno')
                #     # print("[ Top 10 differences ]")
                #     # for stat in top_stats[:10]:
                #     #     print(stat)
                #     # snapshot_prev = snapshot

                if global_step >= ((phase - args.starting_phase)
                                   * (args.mixing_nimg + args.stabilizing_nimg)
                                   + args.mixing_nimg):
                    break

                assert alpha.eval() >= 0

                # if verbose:
                #     writer.flush()

            if verbose:
                print(f"Begin stabilizing epochs in phase {phase}")

            sess.run(assign_zero)

            while True:
                start = time.time()

                # Update learning rate
                d_lr_val = sess.run(update_d_lr)
                g_lr_val = sess.run(update_g_lr)

                assert alpha.eval() == 0
                if global_step % args.checkpoint_every_nsteps == 0 < (batch_size*global_size) and local_step > 0:

                    if args.horovod:
                        sess.run(broadcast)
                    saver = tf.train.Saver(var_list)
                    if verbose:
                        print(f'Writing checkpoint file: model_{phase}_ckpt_{global_step}')
                        saver.save(sess, os.path.join(logdir, f'model_{phase}_ckpt_{global_step}'))

                batch_loc = np.random.randint(0, len(npy_data) - batch_size)
                batch_paths = npy_data[batch_loc: batch_loc + batch_size]
                batch = np.stack([np.load(path) for path in batch_paths])
                batch = batch[:, np.newaxis, ...].astype(np.float32) / 1024 - 1

                small_summary_bool = (local_step % args.summary_small_every_nsteps == 0)
                large_summary_bool = (local_step % args.summary_large_every_nsteps == 0)
                metrics_summary_bool = (local_step % args.metrics_every_nsteps == 0)
                if large_summary_bool:
                    _, _, summary_s, summary_l, d_loss, g_loss = sess.run(
                        [train_gen, train_disc, summary_small, summary_large,
                        disc_loss, gen_loss], feed_dict={real_image_input: batch}) 
                elif small_summary_bool:
                    _, _, summary_s, d_loss, g_loss = sess.run(
                        [train_gen, train_disc, summary_small,
                        disc_loss, gen_loss], feed_dict={real_image_input: batch})
                else:
                    _, _, d_loss, g_loss = sess.run(
                        [train_gen, train_disc, disc_loss, gen_loss], 
                        feed_dict={real_image_input: batch})

#                _, _, d_loss, g_loss = sess.run(
#                        [train_gen, train_disc, disc_loss, gen_loss],
#                        feed_dict={real_image_input: batch})

                global_step += batch_size * global_size
                local_step += 1

                end = time.time()
                local_img_s = batch_size / (end - start)
                img_s = global_size * local_img_s

                sess.run(ema_op)
                in_phase_step = sess.run(update_intra_phase_step)

                if metrics_summary_bool:
                    if args.calc_metrics:
                        # print('Computing and writing metrics')
                        metrics = save_metrics(writer, sess, npy_data, gen_sample, batch_size, global_size, global_step, size, args.horovod, get_compute_metrics_dict(args), num_metric_samples, verbose)

                        # Optuna pruning and return value:
                        last_fid = metrics['FID']
                        if args.optuna_distributed:
                            print(f"Trial: {trial.number}, Worker: {hvd.rank()}, Parameters: {trial.params}, global_step: {global_step}, FID: {last_fid}")
                        else:
                            print(f"Trial: {trial.number}, Parameters: {trial.params}, global_step: {global_step}, FID: {last_fid}")
                        trial.report(metrics['FID'], global_step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                if verbose:

                    if large_summary_bool:
                        print('Writing large summary...')
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='img_s', simple_value=img_s)]), global_step)
                        writer.add_summary(summary_s, global_step)
                        writer.add_summary(summary_l, global_step)
                    elif small_summary_bool:
                        print('Writing small summary...')
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='img_s',
                                                                            simple_value   =img_s)]),
                                        global_step)
                        writer.add_summary(summary_s, global_step)
                    # memory_percentage = psutil.Process(os.getpid()).memory_percent()
                    # if not args.gpu:
                    #     memory_percentage = psutil.Process(os.getpid()).memory_percent()
                    # else:
                    #     gpu_info = nvgpu.gpu_info()
                    #     memory_percentage = nvgpu.gpu_info()[local_rank]['mem_used_percent']

                    # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='memory_percentage', simple_value=memory_percentage)]),
                    #                    global_step)
                    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
                    # print(f"{current_time} \t"
                    #       f"Step {global_step:09} \t"
                    #       f"Step(phase) {in_phase_step:09} \t"
                    #       f"img/s {img_s:.2f} \t "
                    #       f"img/s/worker {local_img_s:.3f} \t"
                    #       f"d_loss {d_loss:.4f} \t "
                    #       f"g_loss {g_loss:.4f} \t "
                    #       f"d_lr {d_lr_val:.5f} \t"
                    #       f"g_lr {g_lr_val:.5f} \t"
                    #       # f"memory {memory_percentage:.4f} % \t"
                    #       f"alpha {alpha.eval():.2f}")


                # if verbose:
                #     writer.flush()

                if global_step >= (phase - args.starting_phase + 1) * (args.stabilizing_nimg + args.mixing_nimg):
                    # if verbose:
                    #     run_metadata = tf.RunMetadata()
                    #     opts = tf.profiler.ProfileOptionBuilder.float_operation()
                    #     g = tf.get_default_graph()
                    #     flops = tf.profiler.profile(g, run_meta=run_metadata, cmd='op', options=opts)
                    #     writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='graph_flops',
                    #                                                           simple_value=flops.total_float_ops)]),
                    #                        global_step)
                    #
                    #     # Print memory info.
                    #     try:
                    #         print(nvgpu.gpu_info())
                    #     except subprocess.CalledProcessError:
                    #         pid = os.getpid()
                    #         py = psutil.Process(pid)
                    #         print(f"CPU Percent: {py.cpu_percent()}")
                    #         print(f"Memory info: {py.memory_info()}")

                    break

            

            if verbose:
                print("\n\n\n End of phase.")

                # Save Session.
                sess.run(ema_update_weights)
                saver = tf.train.Saver(var_list)
                print("Writing final checkpoint file: model_{phase}")
                saver.save(sess, os.path.join(logdir, f'model_{phase}'))

            if args.ending_phase:
                if phase == args.ending_phase:
                    print("Reached final phase, breaking.")
                    break

    return last_fid