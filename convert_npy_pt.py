import numpy as np
import torch
import os
from functools import partial
import glob

root = '/lustre4/2/managed_datasets/LIDC-IDRI/npy/lanczos_3d/'
# root = '/project/davidr/lidc_idri/tfrecords/tfrecords_512x512x512_avg/'
# output = '/project/davidr/lidc_idri/pt/avg/'
output = '/lustre4/2/managed_datasets/LIDC-IDRI/pt/lanczos_3d/'

if not os.path.exists(output):
    os.makedirs(output)

for size in (4, 8, 16, 32, 64, 128, 256, 512):
    folder = os.path.join(root, f'{size}x{size}')
    output_folder = os.path.join(output, f'{size}x{size}')
    print(output_folder)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    shape_z = size
    shape = (size, size, size)

    npys = glob.glob(os.path.join(folder, '*.npy'))	
    
    for i, path in enumerate(npys):
        array = torch.from_numpy(np.load(path).astype(np.int16))
        filename = os.path.join(output_folder, f'{i:04}.pt')
        torch.save(array, filename)
  
