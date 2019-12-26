import glob
import numpy as np
import os
import shutil
import time


class NumpyDataset:
    def __init__(self, npy_dir, scratch_dir, copy_files):
        super(NumpyDataset, self).__init__()
        self.npy_files = glob.glob(npy_dir + '*.npy')
        print(f"Length of dataset: {len(self.npy_files)}")

        if scratch_dir[-1] == '/':
            scratch_dir = scratch_dir[:-1]

        print(scratch_dir)

        if copy_files:
            for f in self.npy_files:
                shutil.copy(f, os.path.join(scratch_dir, f))

        self.scratch_dir = os.path.join(scratch_dir, npy_dir)
        if len(glob.glob(self.scratch_dir)) < len(self.npy_files):
            print(len(glob.glob(os.path.join(scratch_dir, npy_dir))))
            time.sleep(1)

        self.scratch_files = glob.glob(scratch_dir + '*.npy')
        assert len(self.scratch_files) == len(self.npy_files)

        test_npy_array = np.load(self.npy_files[0])[np.newaxis, ...]
        self.shape = test_npy_array.shape
        self.dtype = test_npy_array.dtype
        
    def __iter__(self):
        for path in self.npy_files:
            yield np.load(path)[np.newaxis, ...]

    def __getitem__(self, idx):
        return np.load(self.npy_files[idx])[np.newaxis, ...]
            
    def __len__(self):
        return len(self.npy_files)