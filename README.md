# saraGAN
This repository hosts the 2D and 3D versions of the saraGAN.

## saraGAN 3D

### Dependencies
To run saraGAN 3D you'll need
- TensorFlow 1.14 or 1.15
- Horovod (optional, for multi-GPU or multinode training)
- skimage (install with pip)
- nvgpu (install with pip)

### How to run
- Load any modules or virtual environments that contain the above dependencies.
- Set OMP_NUM_THREADS to a reasonable value. Usually, number-of-cores-per-task - 1 is a reasonable setting (e.g. if you run 2 tasks on a 2-socket CPU node with 2*12 cores, you would set OMP_NUM_THREADS=11 and run 2 tasks on such a node, mapping them by socket)
- export TF_USE_CUDNN=0
- Run multiple tasks with e.g.
mpirun --map-by ppr:1:socket:PE=12 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x TF_USE_CUDNN -x OMP_NUM_THREADS python -u main.py <args>
Some arguments to the main.py are named, but the first three are unnamed:
- python -u main.py [architecture] [dataset_path] [final_shape]
- architecture: one of the architectures in the ../networks/.. folder. E.g. passing 'pgan' will mean using the generator and discrimantor architecture in SURFGAN_3D/networks/pgan
- dataset_path: path to where the dataset can be found. The dataset_path should contain one subdirectory for each of the phases, e.g. 4x4, 8x8, 16x16 etc. Each of those directories contains all of the images, downscaled to that resolution, one file per image, stored as numpy array (e.g. 0001.npy, 0002.npy, etc).
- final_shape: the final shape of the generated images. Used to compute the number of phases.


### Model checkpoints

128x128x32 pgan 'small' model: https://drive.google.com/open?id=1WZ0kiLtDRV8Ac8LdjTD8F7tXXaR8tNq-

256x256x64 pgan 'xs' model: https://drive.google.com/open?id=1GYt1Eqd36cu9-4l7ZfsNNW6C-T0VzQ-0

128x128x32 pgan 'xs' model: https://drive.google.com/open?id=16M6HaaUz0ohuJrpyUb8Ymlfvc6-wZ_3B

64x64x16 pgan 'xs' model: https://drive.google.com/open?id=1tBQ1W9Hj_B-IR1U7zSv4ZWfRSOiwL42n

128x128x32 pgan 'm' model: https://drive.google.com/open?id=14llM6tAxw5wb9NNP0KjkTZiWx_x2fRJl
FID: 209.9393 
