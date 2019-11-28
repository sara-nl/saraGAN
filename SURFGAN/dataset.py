import glob
import numpy as np


class NumpyDataset:
    def __init__(self, npy_dir):
        super(NumpyDataset, self).__init__()
        self.npy_files = glob.glob(npy_dir + '*.npy')
        print(f"Length of dataset: {len(self.npy_files)}")

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
