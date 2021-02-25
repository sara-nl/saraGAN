import h5py
import glob
import argparse
import os
import numpy as np
import multiprocessing

parser = argparse.ArgumentParser(description='Get path to npy dataset')
parser.add_argument('--datadir', type=str, help='Full path to the directory which holds the directories for the different resolutions. In turn, these resolution-specific directories hold the numpy files')
parser.add_argument('--outdir', type=str, help='Full path to store the output. Directory must exist.')
args = parser.parse_args()

resolution_dirs = glob.glob(args.datadir + '*x*')
res_dirs = [res for res in resolution_dirs if os.path.isdir(res)]

print(f"Resolution_dirs: {res_dirs}")

def loop_element(res_dir):
    with h5py.File(f'{os.path.join(args.outdir, os.path.basename(res_dir))}.h5py', 'w-') as f:
        files_in_res = glob.glob(os.path.join(res_dir,'*.npy'))
        # Store each numpy file as a HDF5 dataset:
        for npy_file_path in files_in_res:
            npy_array = np.load(npy_file_path)
            dataset_name = os.path.basename(npy_file_path.replace('.npy', ''))
            print(f"Storing file: {npy_file_path} as HDF5 dataset {dataset_name}")
            f.create_dataset(name=dataset_name, data=npy_array, dtype=npy_array.dtype)

#for res_dir in res_dirs:
a_pool = multiprocessing.Pool()
a_pool.map(loop_element, res_dirs)


        