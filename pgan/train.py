#!/usr/bin/env python
# coding: utf-8

import argparse
from datetime import datetime
import os
import tensorflow as tf
import pickle
import numpy as np
import horovod.tensorflow as hvd
from tensorflow.python.eager import profiler

GPUS = tf.config.experimental.list_physical_devices('GPU')
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)

from network_dev import make_generator, make_discriminator, get_training_functions
from utils import load_lidc_idri_dataset_from_tfrecords, generate_gif, print_gpu_info, print_cpu_info, transform_batch_to_image_grid, save_array_as_gif, nearest_square_root, get_wasserstein_batch, write_wasserstein_distances
from loss import wasserstein_loss, gradient_penalty_loss


def train(generator, 
          discriminator,
          generator_optim,
          discriminator_optim,
          train_generator,
          train_discriminator,
          latent_dim,
          dataset,
          mixing_epochs,
          stabilizing_epochs,
          epoch_size,
          batch_size,
          training_ratio,
          timestamp,
          gradient_penalty_weight,
          learning_rate,
          decay_rate,
          phase,
          summary_writer,
          is_mixing,
          horovod=False):
    
    if is_mixing:
        c_alpha = 1.0
        epoch_range = range(mixing_epochs)
    else:
        c_alpha = 0.0
        epoch_range = range(mixing_epochs, mixing_epochs + stabilizing_epochs)
        
    for epoch in epoch_range:
        discriminator_loss = []
        generator_loss = []    
        
        alpha = tf.fill((batch_size, 1, 1, 1, 1), c_alpha)
        
        if horovod:
            iterator = dataset.take(max(1, epoch_size // hvd.size())) 
            
            # buildup_epochs = 8
            # if epoch == 0:
            #     target_lr = learning_rate.read_value()
            #     learning_rate.assign(0)
            #     buildup_rate = target_lr / buildup_epochs 
            # 
            # elif epoch <= buildup_epochs:
            #     learning_rate.assign_add(buildup_rate)
            
        else:
            iterator = dataset.take(epoch_size) 
            
        for i, image_batch in enumerate(iterator):
            
            is_first_batch = (i == 0)  # and (epoch == 0)
            
            start = datetime.now()
            d_loss = train_discriminator(generator, 
                                         discriminator, 
                                         discriminator_optim, 
                                         batch_size, 
                                         latent_dim, 
                                         image_batch, 
                                         alpha, 
                                         gradient_penalty_weight,
                                         is_first_batch)
            
            raise
            discriminator_loss.append(d_loss)
            
            if i % training_ratio == 0:
                g_loss, x_fake = train_generator(
                    generator,
                    discriminator,
                    generator_optim,
                    batch_size,
                    latent_dim,
                    alpha,
                    is_first_batch
                )
                generator_loss.append(g_loss)
                
                print(x_fake.shape)
                print(image_batch.shape)
                raise
                
            end = datetime.now()
            
               
        if (horovod and hvd.rank() == 0) or not horovod:
            num_images = nearest_square_root(image_batch.shape[0])
            random_indices = np.random.choice(image_batch.shape[0], num_images, replace=False)
            originals = image_batch.numpy()
            fakes = x_fake.numpy()
            image_dir = os.path.join('logs', timestamp)
            original_save_dir = os.path.join(image_dir, 'originals')
            fake_save_dir = os.path.join(image_dir, 'fakes')               
            if not os.path.exists(original_save_dir):
                os.makedirs(original_save_dir)
            if not os.path.exists(fake_save_dir):
                os.makedirs(fake_save_dir)               
            np.save(os.path.join(original_save_dir, f'{epoch}.npy'), originals)
            np.save(os.path.join(fake_save_dir, f'{epoch}.npy'), fakes)
            
            if epoch % 5 == 0:
                generate_gif(fakes[random_indices].squeeze(-1), originals[random_indices].squeeze(-1), image_dir, epoch)
            
            # Write to tensorboard.
            images_per_second = image_batch.shape[0] / (end - start).total_seconds()
            if horovod:
                images_per_second = images_per_second * hvd.size()
            
            
            step = (phase - 1) * (mixing_epochs + stabilizing_epochs) + epoch
            if (horovod and hvd.rank()) == 0 or not horovod:
                with summary_writer.as_default():
                    original = originals[0]
                    fake = fakes[0]                    
                    images_seen_this_epoch = (epoch + 1) * epoch_size * batch_size
                    total_images_seen = (step + 1) * epoch_size * batch_size
                    tf.summary.image("original", original, max_outputs=original.shape[0], step=step)
                    tf.summary.image("fake", original, max_outputs=fake.shape[0], step=step)
                    # tf.summary.scalar("original_min", original.min(), step=step)
                    # tf.summary.scalar("original_max", original.max(), step=step)
                    tf.summary.scalar("fake_min", fake.min(), step=step)
                    tf.summary.scalar("fake_max", fake.max(), step=step)                       
                    tf.summary.scalar("images_seen_this_epoch", images_seen_this_epoch, step=step)
                    tf.summary.scalar("total_images_seen", total_images_seen, step=step)
                    tf.summary.scalar("g_loss", g_loss, step=step)
                    tf.summary.scalar("d_loss", d_loss, step=step)
                    tf.summary.scalar("img/s", images_per_second, step=step)
                    tf.summary.scalar("alpha", c_alpha, step=step)
                    tf.summary.scalar("learning_rate", learning_rate.read_value(), step=step)
                    tf.summary.scalar("time_elapsed_hours", (datetime.now() - datetime.strptime(timestamp, "%Y-%m-%d_%H-%M")).total_seconds() / 3600, step=step)
                    if GPUS:
                        for i, (gpu, memory) in enumerate(print_gpu_info()):
                            tf.summary.scalar(f"{gpu}_{i}", memory, step=step)
 
                    else:
                        pid, memory = print_cpu_info()
                        tf.summary.scalar(f"{pid}", memory, step=step)

        if is_mixing:
            c_alpha -= 1 / mixing_epochs
            
        learning_rate.assign_add(-1 * decay_rate)

    return generator, discriminator

def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = os.path.join('logs', timestamp, 'summaries')
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    num_phases = int(np.log2(args.final_resolution) - 1)
    if (args.horovod and hvd.rank() == 0) or not args.horovod:
        print(f"Number of phases is {num_phases}, final output resolution will be {2 * 2 ** num_phases}")
        
    for phase in range(args.starting_phase, num_phases + 1):
        
        size = 2 * 2 ** phase
       
        tfr_path = args.tfrecords_path + f'tfrecords_{size}x{size}x{size}/'
        # tfr_path = os.path.join(args.tfrecords_path, f'tfrecords_new_lanczos_{size}x{size}')
        
        if not os.path.exists(tfr_path):
            raise ValueError(f"Path doesn't exist: {tfr_path}")
        
        dataset_size = len(os.listdir(tfr_path))
        zdim_phase1 = args.final_zdim // (2 ** ((num_phases - 1)))
        zdim_phase = args.final_zdim // (2 ** ((num_phases - phase)))
        shape = (zdim_phase, size, size, 1)
        
        print(shape)
        
        if args.phase_1_batch_size:
            batch_size = args.phase_1_batch_size // (2 ** phase)
        else:
            batch_size = 512 // size
        batch_size = max(1, batch_size)
        
        dataset = load_lidc_idri_dataset_from_tfrecords(tfr_path, batch_size=batch_size, shape=shape)
        epoch_size = dataset_size // batch_size
        
        generator = make_generator(phase, num_phases, args.base_dim, args.latent_dim, zdim_phase1)
        discriminator = make_discriminator(phase, num_phases, args.base_dim, shape, args.latent_dim)
        
        if (args.horovod and hvd.rank() == 0) or not args.horovod:
            print(f'\n|\t\t\tPhase: {phase} \t Resolution: {size} \t Batch Size: {batch_size} \t Epoch Size: {epoch_size}\t\t\t|\n')
        
        if (args.horovod and hvd.rank() == 0) or not args.horovod:
            print(generator.summary(line_length=120))
            print(discriminator.summary(line_length=120))
        
        if args.horovod:
            learning_rate = tf.Variable(tf.constant(args.learning_rate * np.sqrt(hvd.size())))
            if args.decay_rate == 0:
                decay_rate = (learning_rate.read_value() - args.learning_rate) / (args.mixing_epochs + args.stabilizing_epochs)
            if hvd.rank() == 0:
                print(f"Learning rate: {learning_rate.read_value():.5f} \t Decay rate: {decay_rate:.5f}")
            
            generator_optim = tf.optimizers.Adam(learning_rate, beta_1=0.0, beta_2=0.9)
            discriminator_optim = tf.optimizers.Adam(learning_rate, beta_1=0.0, beta_2=0.9)
            
        else:
            learning_rate = tf.Variable(tf.constant(args.learning_rate))
            generator_optim = tf.optimizers.Adam(learning_rate, beta_1=0.0, beta_2=0.9)
            discriminator_optim = tf.optimizers.Adam(learning_rate, beta_1=0.0, beta_2=0.9)
            decay_rate = args.decay_rate
             
        train_generator, train_discriminator = get_training_functions(args.horovod)
        
        # Tensorboard
        step = (phase - 1) * (args.mixing_epochs + args.stabilizing_epochs)
        checkpoint_path_prev = os.path.join('checkpoints', f'phase_{phase - 1}')
        if (args.horovod and hvd.rank()) == 0 or not args.horovod:
            
            if os.path.exists(checkpoint_path_prev):
                print(f"Loading weights from phase {phase - 1} from {checkpoint_path_prev}")
                # generator.load_weights(os.path.join(checkpoint_path_prev, 'generator.h5'), by_name=True)
                # discriminator.load_weights(os.path.join(checkpoint_path_prev, 'discriminator.h5'), by_name=True)
                
            image_dir = os.path.join('logs', timestamp, f'phase_{phase}_start.gif')
            z = tf.random.normal(shape=(batch_size, args.latent_dim))
            alpha = tf.fill((batch_size, 1, 1, 1, 1), 1.0)
            fakes = generator([z, alpha]).numpy().squeeze(-1)
            num_images = nearest_square_root(batch_size)
            random_indices = np.random.choice(batch_size, num_images, replace=False)       
            img = transform_batch_to_image_grid(fakes[random_indices])
            save_array_as_gif(image_dir, img)           
            
            # if phase >= 2:
            #     with summary_writer.as_default():
            #         real_batch = get_wasserstein_batch(dataset)
            #         z = tf.random.normal(shape=(real_batch.shape[0], args.latent_dim))
            #         alpha = tf.fill((real_batch.shape[0], 1, 1, 1, 1), 1.0)
            #         fake_batch = generator([z, alpha])
            #         write_wasserstein_distances(real_batch, fake_batch, step)
                    
            print("\n\t\t\tStarting mixing epochs\t\t\t\n")
        
            # profiler.start()
    
        generator, discriminator = train(
            generator,
            discriminator,
            generator_optim,
            discriminator_optim,
            train_generator,
            train_discriminator,
            args.latent_dim,
            dataset,
            args.mixing_epochs,
            args.stabilizing_epochs,
            epoch_size,
            batch_size,
            args.training_ratio,
            timestamp,
            args.gradient_penalty_weight,
            learning_rate,
            decay_rate,
            phase,
            summary_writer,
            is_mixing=True,
            horovod=args.horovod
        )
        
        # Tensorboard
        step = (phase - 1) * (args.mixing_epochs + args.stabilizing_epochs) + args.mixing_epochs
        if (args.horovod and hvd.rank()) == 0 or not args.horovod:
            
#             profiler_result = profiler.stop()
#             profiler.save(log_dir, profiler_result)
            # if phase >= 2:
            #     with summary_writer.as_default():
            #         z = tf.random.normal(shape=(real_batch.shape[0], args.latent_dim))
            #         alpha = tf.fill((real_batch.shape[0], 1, 1, 1, 1), 0.0)
            #         fake_batch = generator([z, alpha])
            #         write_wasserstein_distances(real_batch, fake_batch, step=step)
       
            print("\n\t\t\tStarting stabilizing epochs\t\t\t\n")
#             profiler.start()
        
        generator, discriminator = train(
            generator,
            discriminator,
            generator_optim,
            discriminator_optim,
            train_generator,
            train_discriminator,
            args.latent_dim,
            dataset,
            args.mixing_epochs,
            args.stabilizing_epochs,
            epoch_size,
            batch_size,
            args.training_ratio,
            timestamp,
            args.gradient_penalty_weight,
            learning_rate,
            decay_rate,
            phase,
            summary_writer,
            is_mixing=False,
            horovod=args.horovod
        )
        
        
        step = phase * (args.mixing_epochs + args.stabilizing_epochs)
        if (args.horovod and hvd.rank()) == 0 or not args.horovod:
            
#             profiler_result = profiler.stop()
#             profiler.save(log_dir, profiler_result)
            
            image_dir = os.path.join('logs', timestamp, f'phase_{phase}_end.gif')
            z = tf.random.normal(shape=(batch_size, args.latent_dim))
            alpha = tf.fill((batch_size, 1, 1, 1, 1), 0.0)
            fakes = generator([z, alpha]).numpy().squeeze(-1)
            num_images = nearest_square_root(fakes.shape[0])
            random_indices = np.random.choice(fakes.shape[0], num_images, replace=False)              
            img = transform_batch_to_image_grid(fakes[random_indices])
            save_array_as_gif(image_dir, img)       
            
            # if phase >= 2:
            #     with summary_writer.as_default():
            #         z = tf.random.normal(shape=(real_batch.shape[0], args.latent_dim))
            #         alpha = tf.fill((real_batch.shape[0], 1, 1, 1, 1), 0.0)
            #         fake_batch = generator([z, alpha])
            #         write_wasserstein_distances(real_batch, fake_batch, step=step)
       
            checkpoint_path = os.path.join('checkpoints', f'phase_{phase}')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
                
            print(f"\n\nSaving weights to {checkpoint_path}.\n\n")
                
            generator.save_weights(os.path.join(checkpoint_path, 'generator.h5'), save_format='h5')
            discriminator.save_weights(os.path.join(checkpoint_path, 'discriminator.h5'), save_format='h5')
       
        del generator, discriminator, generator_optim, discriminator_optim, train_generator, train_discriminator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tfrecords_path', type=str)
    parser.add_argument('final_resolution', type=int)
    parser.add_argument('final_zdim', type=int)
    parser.add_argument('--starting_phase', type=int, default=1)
    parser.add_argument('--base_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--training_ratio', type=int, default=1)
    parser.add_argument('--phase_1_batch_size', type=int, default=None)
    parser.add_argument('--gradient_penalty_weight', type=int, default=10)
    parser.add_argument('--mixing_epochs', type=int, default=256)
    parser.add_argument('--stabilizing_epochs', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--decay_rate', type=float, default=0)
    parser.add_argument('--horovod', default=False, action='store_true')
    args = parser.parse_args()
    
    if args.horovod:
        hvd.init()
        
        if hvd.rank() == 0:
            print(f"\n\n Using Horovod with global size {hvd.size()} and local size {hvd.local_size()}\n\n")
        
        print(f"Rank {hvd.rank()} reporting!")

        if GPUS:
            tf.config.experimental.set_visible_devices(GPUS[hvd.local_rank()], 'GPU')
        tf.random.set_seed(42 + hvd.rank())

    else:
        print("Not using horovod.")
        tf.random.set_seed(42)
            
    main(args)
