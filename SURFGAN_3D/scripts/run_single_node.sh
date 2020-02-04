#!/usr/bin/bash
cd /home/davidr/projects/saraGAN/SURFGAN/
mpirun -n 2 python main.py /lustre4/2/managed_datasets/LIDC-IDRI/npy/lanczos_3d/ '(1, 128, 512, 512)' --starting_phase=4 --ending_phase=4 --horovod
