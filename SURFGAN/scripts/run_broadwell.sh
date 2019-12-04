#!/usr/bin/bash
cd /home/davidr/projects/saraGAN/SURFGAN/
mpirun -N 2 -npernode 2 python main.py /lustre4/2/managed_datasets/LIDC-IDRI/npy/lanczos_3d/ '(1, 128, 512, 512)' --starting_phase=3 --ending_phase=4 --starting_alpha=1 --calc_metrics --horovod
