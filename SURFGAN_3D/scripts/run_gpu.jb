#!/bin/bash
#SBATCH -N 3
#SBATCH -n 12
#SBATCH -p gpu_titanrtx
#SBATCH -t 5-00:00:00

module use /home/druhe/environment-modules-lisa
module load 2020
module load TensorFlow/1.15.0-foss-2019b-Python-3.7.4-10.1.243

# module purge
# module load 2019
# module load Anaconda3
# module load cuDNN/7.6.5.32-CUDA-10.1.243
# module load NCCL/2.5.6-CUDA-10.1.243
# module load OpenMPI/3.1.4-GCC-7.3.0-2.30
# source activate py37

# . /home/druhe/envs/bin_py36_tf115_hvd018/setup.sh

cd /home/$USER/projects/saraGAN/SURFGAN_3D/
# CONTINUE_PATH=/home/druhe/projects/saraGAN/SURFGAN_3D/runs/pgan2/2020-03-22_09:49:13/model_7_ckpt_98304
# CONTINUE_PATH=/home/druhe/projects/saraGAN/SURFGAN_3D/runs/surfgan/2020-03-18_19:30:52/model_6
#export TF_USE_CUDNN=0
# export TF_CUDNN_USE_AUTOTUNE=1
# 
# mpirun -np 8 -npernode 4 \
#        -bind-to none \
#        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x TF_CUDNN_USE_AUTOTUNE \
#        -mca pml ob1 -mca btl ^openib \
#        python -u main.py pgan2 /nfs/managed_datasets/LIDC-IDRI/npy/average/ '(1, 128, 512, 512)' --starting_phase 7 --ending_phase 7 --latent_dim 512 --horovod --starting_alpha 0 --scratch_path /scratch/$USER --gpu --base_batch_size 32 --network_size s --loss_fn wgan --gp_weight 10 --d_lr 1e-4 --g_lr 1e-3 --continue_path $CONTINUE_PATH


# mpirun -np 16 -npernode 4 \
#     -bind-to none \
#     -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#     -mca pml ob1 -mca btl ^openib \
#     python -u main.py pgan /nfs/managed_datasets/LIDC-IDRI/npy/average/ '(1, 128, 512, 512)' --starting_phase 7 --ending_phase 7 --latent_dim 512 --horovod --starting_alpha 0 --scratch_path /scratch/$USER --gpu --base_batch_size 32 --network_size s --loss_fn wgan --gp_weight 10 --d_lr 1e-3 --g_lr 1e-3
#
# --continue_path $CONTINUE_PATH 
# mpirun -n 4 -npernode 4 python -u main.py pgan2 /nfs/managed_datasets/LIDC-IDRI/npy/average/ '(1, 128, 512, 512)' --scratch_path '/scratch/' --starting_phase 4 --ending_phase 4 --base_dim 512 --latent_dim 512 --horovod --starting_alpha 0 --gpu --base_batch_size=128 --network_size medium --optim_strategy alternate


 mpirun -np 12 -npernode 4 python -u main.py pgan /nfs/managed_datasets/LIDC-IDRI/npy/average/ '(1, 128, 512, 512)' --starting_phase 7 --ending_phase 7 --latent_dim 512 --horovod --starting_alpha 0 --scratch_path /scratch/$USER --gpu --base_batch_size 32 --network_size xs --loss_fn wgan --gp_weight 10 --d_lr 1e-5 --g_lr 1e-5


