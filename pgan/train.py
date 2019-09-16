#!/usr/bin/env python
# coding: utf-8

import argparse
from datetime import datetime
import os
import tensorflow as tf
import pickle
import numpy as np
import horovod.tensorflow as hvd
# from tensorflow.python.eager import profiler

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

from network import make_generator, make_discriminator, get_training_functions
from utils import load_lidc_idri_dataset_from_tfrecords, generate_gif, print_gpu_info, print_cpu_info
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
        if (horovod and hvd.rank() == 0) or not horovod:
            print("Epoch: ", epoch)
            
        discriminator_loss = []
        generator_loss = []    
        
        alpha = tf.fill((batch_size, 1, 1, 1, 1), c_alpha)
        
        if horovod:
            iterator = dataset.take(max(1, epoch_size // hvd.size())) 
        else:
            iterator = dataset.take(epoch_size) 
            
        for i, image_batch in enumerate(iterator):
            
            is_first_batch = (i == 0) and (epoch == 0)

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
            
            end = datetime.now()
            
            if (horovod and hvd.rank() == 0) or not horovod:
                print(f"Processed {image_batch.shape[0]} images in {(end - start).total_seconds()} seconds")
                
        if epoch % 1 == 0:
            if (horovod and hvd.rank() == 0) or not horovod:
                num_images = int(10 - np.log2(image_batch.shape[1])) ** 2
                random_indices = np.random.choice(image_batch.shape[0], num_images)
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
                generate_gif(fakes[random_indices].squeeze(), originals[random_indices].squeeze(), image_dir, epoch)
                
                with summary_writer.as_default():
                    original = originals[0]
                    fake = fakes[0]
                    tf.summary.image("Original example.", original, max_outputs=original.shape[0], step=epoch)
                    tf.summary.image("Fake example.", original, max_outputs=fake.shape[0], step=epoch)

        if is_mixing:
            c_alpha -= 1 / mixing_epochs

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
        
        if not os.path.exists(tfr_path):
            raise ValueError(f"Path doesn't exist: {tfr_path}")
        
        dataset_size = len(os.listdir(tfr_path))
        shape = (size, size, size, 1)
        
        if args.phase_1_batch_size:
            batch_size = args.phase_1_batch_size // size
        else:
            batch_size = 512 // size
        batch_size = max(1, batch_size)
        
        if (args.horovod and hvd.rank() == 0) or not args.horovod:
            print(f'\n|\t\t\tPhase: {phase}, Resolution: {size}, Batch Size: {batch_size}\t\t\t|\n')
        
        dataset = load_lidc_idri_dataset_from_tfrecords(tfr_path, batch_size=batch_size, shape=shape)
        epoch_size = dataset_size // batch_size
        
        generator = make_generator(phase, num_phases, args.base_dim, args.latent_dim)
        discriminator = make_discriminator(phase, num_phases, args.base_dim, shape, args.latent_dim)
        
        if (args.horovod and hvd.rank() == 0) or not args.horovod:
            print(generator.summary())
            print(discriminator.summary())
        
        if args.horovod:
            # Come up with a generalizable approach.
            generator_optim = tf.optimizers.Adam(1e-4 * hvd.size(), beta_1=0.5, beta_2=0.9)
            discriminator_optim = tf.optimizers.Adam(1e-4 * hvd.size(), beta_1=0.5, beta_2=0.9)
            generator_optim = hvd.DistributedOptimizer(generator_optim)
            discriminator_optim = hvd.DistributedOptimizer(discriminator_optim)
        else:
            generator_optim = tf.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.9)
            discriminator_optim = tf.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.9)
             
        train_generator, train_discriminator = get_training_functions(args.horovod)
        
        checkpoint_path_prev = os.path.join('checkpoints', f'phase_{phase - 1}')
        if os.path.exists(checkpoint_path_prev):
            if (args.horovod and hvd.rank() == 0) or not args.horovod:
                print(f"Loading weights from phase {phase - 1} from {checkpoint_path_prev}")
                generator.load_weights(os.path.join(checkpoint_path_prev, 'generator.h5'), by_name=True)
                discriminator.load_weights(os.path.join(checkpoint_path_prev, 'discriminator.h5'), by_name=True)
        
        if (args.horovod and hvd.rank() == 0) or not args.horovod:
            print("\n\t\t\tStarting mixing epochs\t\t\t\n")
            
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
            summary_writer,
            is_mixing=True,
            horovod=args.horovod
        )
        
        if tf.test.gpu_device_name():
            print_fn = print_gpu_info
        else:
            print_fn = print_cpu_info
        
        if (args.horovod and hvd.rank() == 0) or not args.horovod:
            print_fn()
            print("\n\t\t\tStarting stabilizing epochs\t\t\t\n")
        
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
            summary_writer,
            is_mixing=False,
            horovod=args.horovod
        )
        # profiler_result = profiler.stop()
        # profiler.save(log_dir, profiler_result)
        if (args.horovod and hvd.rank() == 0) or not args.horovod:
            print_fn()
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
    parser.add_argument('--starting_phase', type=int, default=1)
    parser.add_argument('--base_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--training_ratio', type=int, default=1)
    parser.add_argument('--phase_1_batch_size', type=int, default=None)
    parser.add_argument('--gradient_penalty_weight', type=int, default=10)
    parser.add_argument('--mixing_epochs', type=int, default=256)
    parser.add_argument('--stabilizing_epochs', type=int, default=256)
    parser.add_argument('--horovod', default=False, action='store_true')
    args = parser.parse_args()
    
    if args.horovod:
        hvd.init()
        
        if hvd.rank() == 0:
            print(f"\n\n Using Horovod with global size {hvd.size()} and local size {hvd.local_size()}\n\n")
        
        print(hvd.rank())
            
    main(args)
