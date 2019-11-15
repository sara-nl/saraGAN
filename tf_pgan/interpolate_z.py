import argparse
import numpy as np
import os
import tensorflow as tf
import horovod.tensorflow as hvd
import time
import random
from resnet import resnet
from metrics.fid import get_fid_for_volumes
from metrics.swd_new_3d import get_swd_for_volumes

from dataset import NumpyDataset
from network import discriminator, generator
from utils import count_parameters, image_grid
from mpi4py import MPI
from tensorflow.data.experimental import AUTOTUNE

import imageio


def main(args, config):
    num_phases = int(np.log2(args.final_resolution) - 1)
    var_list = None
    if args.horovod:
        verbose = hvd.rank() == 0
        global_size = hvd.size()
    else:
        verbose = True
        global_size = 1

    timestamp = time.strftime("%Y-%m-%d_%H:%M", time.gmtime())
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_samples', timestamp)

    if verbose:
        writer = tf.summary.FileWriter(logdir=logdir)
        print("Arguments passed:")
        print(args)
        print(f"Saving files to {logdir}")

    else:
        writer = None

    global_step = 0

    for phase in range(1, num_phases + 1):

        tf.reset_default_graph()
        # Get Dataset.
        size = 2 * 2 ** phase
        data_path = os.path.join(args.dataset_path, f'{size}x{size}/')
        npy_data = NumpyDataset(data_path)
        dataset = tf.data.Dataset.from_generator(npy_data.__iter__, npy_data.dtype, npy_data.shape)

        # Get DataLoader
        if args.base_batch_size:
            batch_size = max(1, args.base_batch_size // (2 ** phase))
        else:
            batch_size = max(1, 128 // size)
        # batch_size = 4

        if args.horovod:
            dataset.shard(hvd.size(), hvd.rank())

        # Lay out the graph.
        real_image_input = dataset. \
            shuffle(len(npy_data)). \
            batch(batch_size, drop_remainder=True). \
            map(lambda x: tf.cast(x, tf.float32) / 1024 - 1, num_parallel_calls=AUTOTUNE). \
            prefetch(AUTOTUNE). \
            repeat(). \
            make_one_shot_iterator(). \
            get_next()

        real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * .01

        with tf.variable_scope('alpha'):
            alpha = tf.Variable(1, name='alpha', dtype=tf.float32)
            # Alpha init

            # Specify alpha update op for mixing phase.
            num_steps = args.mixing_nimg // (batch_size * global_size)
            alpha_update = 1 / num_steps
            update_alpha = alpha.assign(tf.maximum(alpha - alpha_update, 0))

        zdim_base = max(1, args.final_zdim // (2 ** (num_phases - 1)))
        base_shape = (1, zdim_base, 4, 4)

        noise_input_d = tf.placeholder(shape=[real_image_input.shape[0], args.latent_dim], dtype=tf.float32)
        gen_sample_d = generator(noise_input_d, alpha, phase, num_phases,
                                 args.base_dim, base_shape, activation=args.activation, param=args.leakiness)

        disc_fake_d = discriminator(gen_sample_d, alpha, phase, num_phases,
                                    args.base_dim, activation=args.activation, param=args.leakiness, is_reuse=False)

        disc_real_d = discriminator(real_image_input, alpha, phase, num_phases,
                                    args.base_dim, activation=args.activation, param=args.leakiness, is_reuse=True)

        wgan_disc_loss = tf.reduce_mean(disc_fake_d) - tf.reduce_mean(disc_real_d)
        gen_loss = -tf.reduce_mean(disc_fake_d)

        gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
        interpolates = real_image_input + gamma * (tf.stop_gradient(gen_sample_d) - real_image_input)
        gradients = tf.gradients(discriminator(interpolates, alpha, phase,
                                               num_phases, args.base_dim, is_reuse=True, activation=args.activation,
                                               param=args.leakiness), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3, 4)))
        gradient_penalty = tf.reduce_mean((slopes - args.gp_center) ** 2)

        gp_loss = args.gp_weight * gradient_penalty
        drift_loss = 1e-3 * tf.reduce_mean(disc_real_d ** 2)
        disc_loss = wgan_disc_loss + gp_loss + drift_loss

        if args.use_ext_clf:
            real_ext_d = tf.reshape(resnet(real_image_input), (batch_size,))
            fake_ext_d = tf.reshape(resnet(gen_sample_d, is_reuse=True), (batch_size,))

            real_labels = tf.ones(tf.shape(real_ext_d))
            fake_labels = tf.zeros(tf.shape(real_ext_d))

            ext_d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels,
                                                                      logits=real_ext_d)
            ext_d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels,
                                                                      logits=fake_ext_d)

            ext_d_loss = tf.reduce_mean(ext_d_real_loss) + tf.reduce_mean(ext_d_fake_loss)

            ext_d_accuracy_real = tf.keras.metrics.binary_accuracy(real_labels, tf.sigmoid(real_ext_d))
            ext_d_accuracy_fake = tf.keras.metrics.binary_accuracy(fake_labels, tf.sigmoid(fake_ext_d))
            ext_d_accuracy = (ext_d_accuracy_real + ext_d_accuracy_fake) / 2

        if verbose:
            print(f"Generator parameters: {count_parameters('generator')}")
            print(f"Discriminator parameters:: {count_parameters('discriminator')}")
            if args.use_ext_clf:
                print(f"Resnet parameters: {count_parameters('resnet')}")

        # Build Optimizers
        with tf.variable_scope('optim_ops'):

            g_lr = args.learning_rate
            d_lr = args.learning_rate

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

            final_g_lr = g_lr
            final_d_lr = d_lr
            # g_lr_warmup_step = g_lr / args.lr_warmup_epochs
            # d_lr_warmup_step = d_lr / args.lr_warmup_epochs
            g_lr = tf.Variable(final_g_lr, name='g_lr', dtype=tf.float32)  # Start at 0 and warmup.
            d_lr = tf.Variable(final_d_lr, name='d_lr', dtype=tf.float32)
            # warmup_g_lr = g_lr.assign(tf.minimum(final_g_lr, g_lr + g_lr_warmup_step))
            # warmup_d_lr = d_lr.assign(tf.minimum(final_d_lr, d_lr + d_lr_warmup_step))
            anneal_g_lr = g_lr.assign(g_lr * args.g_annealing)
            anneal_d_lr = d_lr.assign(d_lr * args.d_annealing)

            optimizer_gen = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=args.beta1, beta2=args.beta2)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=args.beta1, beta2=args.beta2)

            # Training Variables for each optimizer
            # By default in TensorFlow, all variables are updated by each optimizer, so we
            # need to precise for each one of them the specific variables to update.
            # Generator Network Variables
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            # Discriminator Network Variables
            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            if args.use_ext_clf:
                resnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet')

            if args.horovod:
                optimizer_gen = hvd.DistributedOptimizer(optimizer_gen)
                optimizer_disc = hvd.DistributedOptimizer(optimizer_disc)

            # Create training operations
            train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
            train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)
            if args.use_ext_clf:
                res_lr = 1e-3
                if args.horovod:
                    res_lr = res_lr * hvd.size()
                optimizer_resnet = tf.train.AdamOptimizer(learning_rate=res_lr)
                if args.horovod:
                    optimizer_resnet = hvd.DistributedOptimizer(optimizer_resnet)
                train_resnet = optimizer_resnet.minimize(ext_d_loss, var_list=resnet_vars)

        with tf.name_scope('summaries'):
            # Summaries
            tf.summary.scalar('d_loss', disc_loss)
            tf.summary.scalar('g_loss', gen_loss)
            tf.summary.scalar('gp', gp_loss)
            if args.use_ext_clf:
                tf.summary.scalar('ext_d_loss', ext_d_loss)

            real_image_grid = tf.transpose(real_image_input[0], (1, 2, 3, 0))
            shape = real_image_grid.get_shape().as_list()
            grid_cols = int(2 ** np.floor(np.log(np.sqrt(shape[0])) / np.log(2)))
            grid_rows = shape[0] // grid_cols
            grid_shape = [grid_rows, grid_cols]
            real_image_grid = image_grid(real_image_grid, grid_shape, image_shape=shape[1:3],
                                         num_channels=shape[-1])

            fake_image_grid = tf.transpose(gen_sample_d[0], (1, 2, 3, 0))
            fake_image_grid = image_grid(fake_image_grid, grid_shape, image_shape=shape[1:3],
                                         num_channels=shape[-1])

            tf.summary.image('real_image', real_image_grid)
            tf.summary.image('fake_image', fake_image_grid)

            tf.summary.scalar('fake_image_min', tf.math.reduce_min(gen_sample_d))
            tf.summary.scalar('fake_image_max', tf.math.reduce_max(gen_sample_d))

            tf.summary.scalar('real_image_min', tf.math.reduce_min(real_image_input[0]))
            tf.summary.scalar('real_image_max', tf.math.reduce_max(real_image_input[0]))
            tf.summary.scalar('alpha', alpha)

            tf.summary.scalar('g_lr', g_lr)
            tf.summary.scalar('d_lr', d_lr)

            merged_summaries = tf.summary.merge_all()

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())

            print(phase, args.starting_phase)

            if var_list is not None and phase > args.starting_phase:
                var_names = [v.name for v in var_list]
                trainable_variable_names = [v.name for v in tf.trainable_variables()]
                load_vars = [sess.graph.get_tensor_by_name(n) for n in var_names if n in trainable_variable_names]
                saver = tf.train.Saver(load_vars)
                if verbose:
                    print(f"Restoring session with {var_names} variables.")
                    saver.restore(sess, os.path.join(logdir, f'model_{phase - 1}'))

            elif var_list is not None and args.continue_path and phase == args.starting_phase:
                var_list = gen_vars + disc_vars
                var_names = [v.name for v in var_list]
                trainable_variable_names = [v.name for v in tf.trainable_variables()]
                load_vars = [sess.graph.get_tensor_by_name(n) for n in var_names if n in trainable_variable_names]
                saver = tf.train.Saver(load_vars)
                if verbose:
                    print(f"Restoring session with {var_names} variables.")
                    saver.restore(sess, args.continue_path)

            var_list = gen_vars + disc_vars

            if phase < args.starting_phase:
                continue

            if args.horovod:
                sess.run(hvd.broadcast_global_variables(0))


            samples = []

            z_a = np.random.randn(*noise_input_d.get_shape().as_list())
            for _ in range(args.num_samples):
                z_b = np.random.randn(*noise_input_d.get_shape().as_list())

                linspace = np.linspace(0, 1, 8)

                for p in linspace:
                    print(p)
                    z = (1 - p) * z_a + p * z_b
                    grid = sess.run(
                        fake_image_grid,
                        feed_dict={noise_input_d: z}
                    )

                    samples.append(np.squeeze(grid))

                z_a = z_b

            def normalize(x, logical_minimum, logical_maximum):
                x = x.astype(np.float32)
                x = np.clip(x, logical_minimum, logical_maximum)
                x = (x - x.min()) / (x.max() - x.min())  # [0, 1]
                assert x.min() >= 0 and x.max() <= 1
                x = (x * 255).astype(np.uint8)
                return x

            vid = normalize(np.stack(samples).squeeze(), logical_minimum=-1, logical_maximum=2)
            print(vid.shape)
            np.save('latent_space.npy', vid)
            imageio.mimwrite('videos/latent_space.avi', vid, fps=15)

            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('final_resolution', type=int)
    parser.add_argument('final_zdim', type=int)
    parser.add_argument('num_samples', type=int)
    parser.add_argument('--starting_phase', type=int, default=1)
    parser.add_argument('--ending_phase', type=int, default=None)
    parser.add_argument('--base_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--base_batch_size', type=int, default=None)
    parser.add_argument('--mixing_nimg', type=int, default=2 ** 17)
    parser.add_argument('--stabilizing_nimg', type=int, default=2 ** 17)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--gp_center', type=float, default=1)
    parser.add_argument('--gp_weight', type=float, default=10)
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horovod', default=False, action='store_true')
    parser.add_argument('--fp16_allreduce', default=False, action='store_true')
    parser.add_argument('--calc_metrics', default=False, action='store_true')
    parser.add_argument('--use_ext_clf', default=False, action='store_true')
    parser.add_argument('--g_annealing', default=1,
                        type=float, help='generator annealing rate, 1 -> no annealing.')
    parser.add_argument('--d_annealing', default=1,
                        type=float, help='discriminator annealing rate, 1 -> no annealing.')
    parser.add_argument('--num_metric_samples', type=int, default=512)
    parser.add_argument('--beta1', type=float, default=0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--d_scaling', default='none', choices=['linear', 'sqrt', 'none'],
                        help='How to scale discriminator learning rate with horovod size.')
    parser.add_argument('--g_scaling', default='none', choices=['linear', 'sqrt', 'none'],
                        help='How to scale generator learning rate with horovod size.')
    parser.add_argument('--continue_path', default=None, type=str)
    # parser.add_argument('--lr_warmup_epochs', default=5, type=int)
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.horovod:
        hvd.init()
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
        os.environ['KMP_BLOCKTIME'] = str(1)
        os.environ['OMP_NUM_THREADS'] = str(16)

        np.random.seed(args.seed + hvd.rank())
        tf.random.set_random_seed(args.seed + hvd.rank())
        random.seed(args.seed + hvd.rank())

        print(f"Rank {hvd.rank()} reporting!")

    else:
        np.random.seed(args.seed)
        tf.random.set_random_seed(args.seed)
        random.seed(args.seed)

    main(args, config)
