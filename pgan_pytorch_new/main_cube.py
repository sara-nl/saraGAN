import torch
import horovod.torch as hvd
import numpy as np
import random
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from data import DatasetFolder
from network_o import Generator, Discriminator
from utils import count_parameters
from train import train
from distutils.dir_util import copy_tree
import time
import glob


def main(args):
    num_phases = int(np.log2(args.final_resolution) - 1)
    if (args.horovod and hvd.rank() == 0) or not args.horovod:
        print(f"Number of phases is {num_phases}, "
              f"final output resolution will be {2 * 2 ** num_phases}")
        
        verbose = True
        writer = SummaryWriter()
        
    else:
        writer = None
        verbose = False
        
    # Get Networks
    zdim_base = max(1, args.final_zdim // (2 ** ((num_phases - 1))))
    print(zdim_base)
    generator = Generator(args.starting_phase, num_phases, 
                          args.base_dim, args.latent_dim, (1, zdim_base, 4, 4))
    discriminator = Discriminator(args.starting_phase, num_phases, 
                          args.base_dim, args.latent_dim, (1, zdim_base, 4, 4))
        
    for phase in range(args.starting_phase, num_phases + 1):
        
        if phase  > args.starting_phase:
            
            # Prevents horovod error.
            generator = deepcopy(generator)
            discriminator = deepcopy(discriminator)
            
            generator.grow()
            discriminator.grow()
            
        if verbose:
            print(generator)
            print(f"Number of Generator parameters: {count_parameters(generator)}")
            print(discriminator)
            print(f"Number of Discriminator parameters: {count_parameters(discriminator)}")
        
        # Get Dataset.
        size = 2 * 2 ** phase
        data_path = os.path.join(args.dataset_path, f'{size}x{size}x{size}/')
        scratch_path = os.path.join('/scratch/', f'{size}x{size}x{size}')
        if (args.horovod and hvd.local_rank() == 0) or not args.horovod:
            print("Copying files to scratch space.")
            copy_tree(data_path, scratch_path, preserve_symlinks=True, update=True)
            print('Done!')
            
        while len(glob.glob(os.path.join(scratch_path, '*.pt'))) < len(glob.glob(os.path.join(data_path, '*.pt'))):
            print(hvd.local_rank(), len(glob.glob(os.path.join(scratch_path, '*.pt'))), len(glob.glob(os.path.join(data_path, '*.pt'))))
            time.sleep(1)
        
        assert len(glob.glob(os.path.join(scratch_path, '*.pt'))) == len(glob.glob(os.path.join(data_path, '*.pt')))

        dataset = DatasetFolder(scratch_path, 
                               loader=lambda path: torch.load(path),
                               extensions=('pt',),
                               # transform=lambda x: x.unsqueeze(0).float() / 1024
                               transform=lambda x: x.float() 
        )

        assert len(dataset) == len(glob.glob(os.path.join(scratch_path, '*.pt')))
        
        print('Dataset len, min, max, shape:', len(dataset), dataset[0].min(), dataset[0].max(), dataset[0].shape)
        
        # Get DataLoader
        if args.base_batch_size:
            batch_size = max(1, args.base_batch_size // (2 ** phase))
        else:
            batch_size = max(1, 128 // size)
            
        if args.horovod:
            verbose = hvd.rank() == 0
            torch.set_num_threads(8)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank())
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size,
            sampler=train_sampler, num_workers=8, pin_memory=True)

        else:
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            train_sampler = None
        
        # Get Optimizers
        lr_d = 1e-3
        lr_g = 1e-3
        if args.horovod:
            lr_d = lr_d * np.sqrt(hvd.size())
            lr_g = lr_g * np.sqrt(hvd.size())
        d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0, 0.9))
        g_optim = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0, 0.9))
        if args.horovod:
            # Horovod: (optional) compression algorithm.
            compression = hvd.Compression.fp16 if \
                args.fp16_allreduce else hvd.Compression.none
            
            # Horovod: wrap optimizer with DistributedOptimizer.
            g_optim = hvd.DistributedOptimizer(
                g_optim, named_parameters=generator.named_parameters(),
                compression=compression)
            
            d_optim = hvd.DistributedOptimizer(
                d_optim, named_parameters=discriminator.named_parameters(),
                compression=compression,
                backward_passes_per_step=4)
            
        if (args.horovod and verbose) or not args.horovod:
            print(f'\n|\t\tPhase: {phase} \t Resolution: {size}' 
                  f'\tBatch Size: {batch_size} \t Epoch Size: {len(data_loader)}\t\t|\n')
        
        # Start training.
        train(generator, 
              discriminator, 
              g_optim, 
              d_optim, 
              data_loader,
              args.mixing_epochs, 
              args.stabilizing_epochs,
              phase,
              writer,
              args.horovod,
        )
        
        # Save models.
        if (args.horovod and hvd.rank() == 0) or not args.horovod:
            discriminator.eval()
            generator.eval()
            torch.save(discriminator.state_dict(), os.path.join(writer.log_dir, f'discriminator_phase_{phase}.pt'))
            torch.save(generator.state_dict(), os.path.join(writer.log_dir, f'generator_phase_{phase}.pt'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('final_resolution', type=int)
    parser.add_argument('final_zdim', type=int)
    parser.add_argument('--starting_phase', type=int, default=2)
    parser.add_argument('--base_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--base_batch_size', type=int, default=None)
    parser.add_argument('--mixing_epochs', type=int, default=256)
    parser.add_argument('--stabilizing_epochs', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horovod', default=False, action='store_true')
    parser.add_argument('--fp16_allreduce', default=False, action='store_true')
    args = parser.parse_args()
    
    if args.horovod:
        hvd.init()
        seed = args.seed + hvd.rank()
        if hvd.rank() == 0:
            print(f"\n\n Using Horovod with global size {hvd.size()}"
                  f"and local size {hvd.local_size()}\n\n")
        
        print(f"Rank {hvd.rank()} reporting!")

        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())

    else:
        seed = args.seed
        print("Not using horovod.")
        
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    main(args)
