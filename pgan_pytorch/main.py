import torch
import horovod.torch as hvd
import numpy as np
import random
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

from data import DatasetFolder
from network_ours import Generator, Discriminator
from utils import count_parameters
from train import train


def main(args):
    num_phases = int(np.log2(args.final_resolution) - 1)
    if (args.horovod and hvd.rank() == 0) or not args.horovod:
        print(f"Number of phases is {num_phases},"
              f"final output resolution will be {2 * 2 ** num_phases}")
        
        writer = SummaryWriter()
    else:
        writer = None
        
    for phase in range(args.starting_phase, num_phases + 1):
        # Get Dataset.
        size = 2 * 2 ** phase
        data_path = os.path.join(args.dataset_path, f'{size}x{size}/')
        dataset = DatasetFolder(data_path, 
                               loader=lambda path: torch.load(path),
                               extensions=('pt',),
                               transform=lambda x: x.unsqueeze(0).float() / 1024)
        
        # Get DataLoader
        if args.base_batch_size:
            batch_size = max(1, args.base_batch_size // (2 ** phase))
        else:
            batch_size = max(1, 128 // size)
            
        if args.horovod:
            verbose = hvd.rank() == 0
            torch.set_num_threads(4)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank())
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size,
            sampler=train_sampler, num_workers=4, pin_memory=True)

        else:
            verbose = True
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            train_sampler = None
        
        # Get Networks
        zdim_base = max(1, args.final_zdim // (2 ** ((num_phases - 1))))
        generator = Generator(phase, num_phases, 
                              args.base_dim, args.latent_dim, (1, zdim_base, 4, 4))
        discriminator = Discriminator(phase, num_phases, 
                              args.base_dim, args.latent_dim, (1, zdim_base, 4, 4))
        
        if verbose:
            print(generator)
            print(f"Number of Generator parameters: {count_parameters(generator)}")
            print(discriminator)
            print(f"Number of Discriminator parameters: {count_parameters(discriminator)}")
        
        # Load weights from previous phase.
        if (args.horovod and hvd.rank() == 0) or not args.horovod:
            # Load weights from previous phase.
            discriminator_dir = os.path.join(writer.log_dir, f'discriminator_phase_{phase - 1}.pt')
            generator_dir = os.path.join(writer.log_dir, f'generator_phase_{phase - 1}.pt')
            if os.path.exists(discriminator_dir) and os.path.exists(generator_dir):
                discriminator.eval()
                generator.eval()
                print(f"Loading weights from phase {phase - 1}")
                inc_keys_discriminator = discriminator.load_state_dict(torch.load(discriminator_dir), strict=False)
                # This is dependent on architecture, but I keep it in for safety reasons for now.
                assert len(inc_keys_discriminator[0]) == 6  
                inc_keys_generator = generator.load_state_dict(torch.load(generator_dir), strict=False)
                assert len(inc_keys_generator[0]) == 6

        # Get Optimizers
        lr_d = 2e-3
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
                compression=compression)
            
        
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
    parser.add_argument('--starting_phase', type=int, default=1)
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
