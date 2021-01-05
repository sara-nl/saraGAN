import glob
import numpy as np
import os
import shutil
import time
import tensorflow as tf
import multiprocessing
import itertools
import psutil

def stdnormal_to_8bit_numpy(normalized_input, verbose):
    """Maps standard normalized channels (mean=0, stddev=1) to 8-bit channels ([0,255]).
    Mapping is linear, with the mean (0) being mapped to 128, mean - 2*SD (i.e. -2) being mapped to 0, and mean +2*SD(i.e. 2) being mapped to 256.
    Then, clipping is done to the [0, 255] range, and conversion to integers is performed.
    Parameters:
        normalized_input: a normalized input [np.array] with mean 0 and stddev 1
        verbose: print verbose output [bool]
    returns:
        image_8bit: an integer image with data range [0,255]. Note that the data type is not really unsigned int8, but [int]
    """
    image_8bit = np.clip((64 * normalized_input + 128), 0, 255)
    return image_8bit.astype(int)

def normalize(unnormalized_input, mean, stddev, verbose):
    """(Standard) Normalizes the input data as part of the tensorflow graph. Normalization is done by subtracting the mean, then dividing by the standard deviation.
    Parameters:
        unnormalized_input: the (unnormalized) input data [tf.Tensor]
        mean: the mean of the dataset [float]
        stddev: the standard deviation of the dataset [float]
        verbose: print verbose output [bool]
    returns:
        normalized_input: the input data after normalization [tf.Tensor]
    """
    if mean is None and stddev is None:
        print("INFO: no data_mean or data_stddev was defined, not normalizing the input data")
        return unnormalized_input
    elif mean is None and stddev is not None:
        raise Exception("ERROR: data_stddev was defined, but data_mean was not. Either define both to apply input normalization, or define neither to not apply input normalization")
    elif mean is not None and stddev is None:
        raise Exception("ERROR: data_mean was defined, but data_stddev was not. Either define both to apply input normalization, or define neither to not apply input normalization")
    else:
        with tf.variable_scope('input_normalization', reuse=tf.AUTO_REUSE):
            mean_tensor = tf.get_variable("data_mean", dtype = tf.float32, trainable=False, initializer = mean)
            stddev_tensor = tf.get_variable("data_stddev", dtype = tf.float32, trainable=False, initializer = stddev)
            x = tf.math.subtract(unnormalized_input, mean_tensor)
            normalized_input = tf.math.divide(x, stddev_tensor)
            return normalized_input

def invert_normalize(normalized_input, mean, stddev, verbose):
    """Invert normalization as done by normalize. Operations to undo normalization become part of the tensorflow graph. Normalization is done by subtracting the mean, then dividing by the standard deviation.
    Parameters:
        normalized_input: the normalized input data [tf.Tensor]
        mean: the mean of the dataset [float]
        stddev: the standard deviation of the dataset [float]
        verbose: print verbose output [bool]
    returns:
        unnormalized_input: the input data after inverting the normalization [tf.Tensor]
    """
    if mean is None and stddev is None:
        print("INFO: no data_mean or data_stddev was defined, not inverting normalizing the input data")
        return normalized_input
    elif mean is None and stddev is not None:
        raise Exception("ERROR: data_stddev was defined, but data_mean was not. Either define both to apply input normalization, or define neither to not apply input normalization")
    elif mean is not None and stddev is None:
        raise Exception("ERROR: data_mean was defined, but data_stddev was not. Either define both to apply input normalization, or define neither to not apply input normalization")
    else:
        with tf.variable_scope('input_normalization', reuse=tf.AUTO_REUSE):
            mean_tensor = tf.get_variable("data_mean", dtype = tf.float32, trainable=False, initializer = mean)
            stddev_tensor = tf.get_variable("data_stddev", dtype = tf.float32, trainable=False, initializer = stddev)
            x = tf.math.multiply(normalized_input, stddev_tensor)
            unnormalized_input = tf.math.add(x, mean_tensor)
            return unnormalized_input

def normalize_numpy(unnormalized_input, mean, stddev, verbose):
    """Numpy equivalent of Normalize. Performs standard normalization on the input by subtracting the mean, then dividing by the standard devation.
    Parameters:
        unnormalized_input: the (unnormalized) input data [np.array]
        mean: the mean of the dataset [float]
        stddev: the standard deviation of the dataset [float]
        verbose: print verbose output [bool]
    returns:
        normalized_input: the input data after normalization [np.array]
    """
    if mean is None and stddev is None:
        print("INFO: no data_mean or data_stddev was defined, not normalizing the input data")
        return unnormalized_input
    elif mean is None and stddev is not None:
        raise Exception("ERROR: data_stddev was defined, but data_mean was not. Either define both to apply input normalization, or define neither to not apply input normalization")
    elif mean is not None and stddev is None:
        raise Exception("ERROR: data_mean was defined, but data_stddev was not. Either define both to apply input normalization, or define neither to not apply input normalization")
    else:
        normalized_input = (unnormalized_input - mean) / stddev
        return normalized_input

def invert_normalize_numpy(normalized_input, mean, stddev, verbose):
    """Invert the numpy normalization that is performed by normalize_numpy().
    Parameters:
        normalized_input: the (normalized) input data [np.array]
        mean: the mean of the dataset [float]
        stddev: the standard deviation of the dataset [float]
        verbose: print verbose output [bool]
    returns:
        unnormalized_input: the input data after inverting the normalization [np.array]
    """
    if mean is None and stddev is None:
        print("INFO: no data_mean or data_stddev was defined, not inverting normalizing of the input data")
        return normalized_input
    elif mean is None and stddev is not None:
        raise Exception("ERROR: data_stddev was defined, but data_mean was not. Either define both to apply input normalization, or define neither to not apply input normalization")
    elif mean is not None and stddev is None:
        raise Exception("ERROR: data_mean was defined, but data_stddev was not. Either define both to apply input normalization, or define neither to not apply input normalization")
    else:
        unnormalized_input = (normalized_input * stddev) + mean
        return unnormalized_input

class NumpyDataset:
    def __init__(self, npy_dir, scratch_dir, copy_files, is_correct_phase):
        super(NumpyDataset, self).__init__()
        self.npy_files = glob.glob(npy_dir + '*.npy')
        print(f"Length of dataset: {len(self.npy_files)}")

        if scratch_dir is not None:
            if scratch_dir[-1] == '/':
                scratch_dir = scratch_dir[:-1]

        self.scratch_dir = os.path.normpath(scratch_dir + npy_dir) if is_correct_phase else npy_dir
        if copy_files and is_correct_phase:
            os.makedirs(self.scratch_dir, exist_ok=True)
            print("Copying files to scratch...")
            for f in self.npy_files:
                # os.path.isdir(self.scratch_dir)
                if not os.path.isfile(os.path.normpath(scratch_dir + f)):
                    shutil.copy(f, os.path.normpath(scratch_dir + f))

        while len(glob.glob(self.scratch_dir + '/*.npy')) < len(self.npy_files):
            time.sleep(1)

        self.scratch_files = glob.glob(self.scratch_dir + '/*.npy')
        assert len(self.scratch_files) == len(self.npy_files)

        test_npy_array = np.load(self.npy_files[0])[np.newaxis, ...]
        self.shape = test_npy_array.shape
        self.dtype = test_npy_array.dtype
        del test_npy_array

    def __iter__(self):
        for path in self.npy_files:
            yield np.load(path)[np.newaxis, ...]

    def __getitem__(self, idx):
        return np.load(self.npy_files[idx])[np.newaxis, ...]

    def __len__(self):
        return len(self.npy_files)


class NumpyPathDataset:
    def __init__(self, npy_dir, scratch_dir, copy_files, is_correct_phase):
        super(NumpyPathDataset, self).__init__()
        self.npy_files = glob.glob(npy_dir + '*.npy')
        print(f"Length of dataset: {len(self.npy_files)}")

        if scratch_dir is not None:
            if scratch_dir[-1] == '/':
                scratch_dir = scratch_dir[:-1]

        self.scratch_dir = os.path.normpath(scratch_dir + npy_dir) if is_correct_phase else npy_dir
        if copy_files and is_correct_phase:
            os.makedirs(self.scratch_dir, exist_ok=True)
            print("Copying files to scratch...")
            for f in self.npy_files:
                # os.path.isdir(self.scratch_dir)
                if not os.path.isfile(os.path.normpath(scratch_dir + f)):
                    shutil.copy(f, os.path.normpath(scratch_dir + f))

        while len(glob.glob(self.scratch_dir + '/*.npy')) < len(self.npy_files):
            time.sleep(1)

        self.scratch_files = glob.glob(self.scratch_dir + '/*.npy')
        assert len(self.scratch_files) == len(self.npy_files)

        test_npy_array = np.load(self.scratch_files[0])[np.newaxis, ...]
        self.shape = test_npy_array.shape
        self.dtype = test_npy_array.dtype
        del test_npy_array

    def __iter__(self):
        for path in self.scratch_files:
            yield path

    def __getitem__(self, idx):
        return self.scratch_files[idx]

    def __len__(self):
        return len(self.scratch_files)

