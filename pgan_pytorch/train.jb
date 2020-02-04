#!/bin/bash
#SBATCH -p gpu_titanrtx
#SBATCH -t 120:00:00
#SBATCH -N 1
#SBATCH -n 4
source /home/davidr/scripts/nki_torch.sh

JOBSPERNODE=4

# Send node names to array.
NODELIST=$(scontrol show hostname | paste -d, -s)
IFS=',' read -ra NODE_ARRAY <<< "$NODELIST"
HOSTS=""
for NODE in "${NODE_ARRAY[@]}"; do
    echo $NODE; 
    HOSTS="${HOSTS}$NODE:$JOBSPERNODE,"; 
done
HOSTS=${HOSTS::-1}

echo ${HOSTS}  

# echo "Copying files to scratch."
# DATASET_DIR='/project/davidr/lidc_idri/pt/lanczos_3d/'
# DATASET_DIR='/nfs/managed_datasets/LIDC-IDRI/pt/avg/'
DATASET_DIR='/nfs/managed_datasets/LIDC-IDRI/pt/average_no_pad_/'
# DATASET_DIR='/nfs/managed_datasets/LIDC-IDRI/pt/lanczos_3d/'
# SCRATCH_DIR='/scratch/lanczos_3d'
# rsync -r --progress $DATASET_DIR $SCRATCH_DIR

# horovodrun -np $SLURM_NTASKS -H ${HOSTS} --start-timeout 120 --verbose python main.py $DATASET_DIR 512 128 --base_dim 256 --latent_dim 256 --mixing_epochs 128 --stabilizing_epochs 128 --starting_phase 2 --horovod
# horovodrun -np $SLURM_NTASKS -H ${HOSTS} --start-timeout 120 --verbose python main.py $DATASET_DIR 512 512 --base_dim 256 --latent_dim 256 --mixing_epochs 128 --stabilizing_epochs 128 --starting_phase 2 --horovod
# horovodrun -np $SLURM_NTASKS -H ${HOSTS} --start-timeout 120 --verbose python main.py $DATASET_DIR 512 128 --base_dim 256 --latent_dim 256 --mixing_epochs 128 --stabilizing_epochs 128 --starting_phase 2 --horovod --use_swish --amsgrad
horovodrun -np $SLURM_NTASKS -H ${HOSTS} --start-timeout 120 --verbose python main.py $DATASET_DIR 512 128 --base_dim 256 --latent_dim 256 --mixing_epochs 128 --stabilizing_epochs 128 --starting_phase 6 --horovod --continue_path /home/davidr/projects/saraGAN/pgan_pytorch_new/runs/Oct21_10-08-46_r34n1.lisa.surfsara.nl/

# mpirun -N $SLURM_NNODES -npernode 4 python main.py /project/davidr/lidc_idri/pt/absmax/ 512 128 --base_dim 256 --latent_dim 256 --mixing_epochs 256 --stabilizing_epochs 256 --starting_phase 2 --horovod
