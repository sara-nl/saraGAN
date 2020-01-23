#!/usr/bin/bash
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -p broadwell
#SBATCH -t 0-00:01:00

scontrol update JobId=$SLURM_JOB_ID TimeLimit=1-00:00:00
# . /home/davidr/envs/py36_tf114_hvd019/setup.sh
. /home/davidr/tf_cpu36.sh
cd /home/davidr/projects/saraGAN/SURFGAN/

export KMP_BLOCKTIME=0
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export KMP_SETTINGS=TRUE
# export OMP_NUM_THREADS=8
export OMP_NUM_THREADS=16

DIM=512

python -u main.py stylegan2 /lustre4/2/managed_datasets/LIDC-IDRI/npy/average/ '(1, 128, 512, 512)' --scratch_path '/' --starting_phase 5 --ending_phase 5 --base_dim $DIM --latent_dim $DIM --horovod --starting_alpha 0 --base_batch_size 8 --max_global_batch_size 1024 --learning_rate 0.001
# mpirun -np 64 -ppn 2 --map-by ppr:1:node:pe=8 python -u main.py pgan2 /lustre4/2/managed_datasets/LIDC-IDRI/npy/average/ '(1, 128, 512, 512)' --scratch_path '/' --starting_phase 6 --ending_phase 6 --base_dim 1024 --latent_dim 1024 --horovod --starting_alpha 0 --base_batch_size 64 --max_global_batch_size 1024 --learning_rate 0.001
