import glob
import numpy as np
import os
import shutil
import time
import tensorflow as tf
import multiprocessing
import itertools
import psutil
import copy
import random

from mpi4py import MPI

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


# TODO: create init based on filelist (a glob-like object). This way, we can easily split a glob-like object into training/validation/test parts,
# and create training/validation/test NumpyPathDataset for each of those.
class NumpyPathDataset:
    def __init__(self, npy_dir, scratch_dir, copy_files, is_correct_phase):
        super(NumpyPathDataset, self).__init__()
        self.npy_files = glob.glob(npy_dir + '*.npy')
        print(f"Length of dataset: {len(self.npy_files)}")

        if scratch_dir is not None:
            if scratch_dir[-1] == '/':
                scratch_dir = scratch_dir[:-1]

        self.scratch_dir = os.path.normpath(scratch_dir + npy_dir) if is_correct_phase else npy_dir
        self._copy_files_to_scratch(scratch_dir, copy_files, is_correct_phase)

        while len(glob.glob(self.scratch_dir + '/*.npy')) < len(self.npy_files):
            time.sleep(1)

        self.scratch_files = glob.glob(self.scratch_dir + '/*.npy')
        assert len(self.scratch_files) == len(self.npy_files)

        # Initialize the samplebuffer
        self._init_samplebuffer()

        test_npy_array = np.load(self.scratch_files[0])[np.newaxis, ...]
        self.shape = test_npy_array.shape
        self.dtype = test_npy_array.dtype
        del test_npy_array

        # TODO: Split the dataset into a test and training set (add arguments to __init__ to determine the fractions, then use those in get_training_batch to randomly select samples from the first X% of the dataset)

    def _copy_files_to_scratch(self, scratch_dir, copy_files, is_correct_phase):
        if copy_files and is_correct_phase:
            print(f"Scratch dir: {self.scratch_dir}")
            os.makedirs(self.scratch_dir, exist_ok=True)
            print("Copying files to scratch...")
            for f in self.npy_files:
                # os.path.isdir(self.scratch_dir)
                if not os.path.isfile(os.path.normpath(scratch_dir + f)):
                    shutil.copy(f, os.path.normpath(scratch_dir + f))

    def _init_samplebuffer(self):
        # Note: the [:] on self.scratch_files is needed to make sure the list gets duplicated - otherwise self.scratch_files will also get shuffled
        self.samplebuffer = self.scratch_files[:]
        random.shuffle(self.samplebuffer)

    def __iter__(self):
        for path in self.scratch_files:
            yield path

    def __getitem__(self, idx):
        return self.scratch_files[idx]

    def __len__(self):
        return len(self.scratch_files)

    def split_by_fraction(self, fraction):
        """Split this NumpyPathDataset object into multiple NumpyPathDataset objects, according to provided ratios. E.g. for creating a train, validation and test set.
        Parameters:
            fraction: fraction according to which to split the dataset. E.g. 0.7 will return a one dataset with 70% of the original samples, and another with 30%.
        Returns:
            dataset1, dataset2: two NumpyPathDataset objects
        """

        nsamples_dataset1 = int( np.round(fraction*len(self.scratch_files)) + 1e-5)
        nsamples_dataset2 = len(self.scratch_files)

        # If the number of computed samples for either dataset isn't at least 1, something most likely went wrong
        assert nsamples_dataset1 > 0 and nsamples_dataset2 > 0

        return self.split_by_index(nsamples_dataset1)
    
    def split_by_index(self, index):
        """Split this NumpyPathDataset object into multiple NumpyPathDataset objects, according to provided index. E.g. for creating a train, validation and test set.
        Parameters:
            index: index to the self.scratch_files array that will determine the last sample that is part of dataset1. index+1 will be the first sample of dataset2.
        Returns:
            dataset1, dataset2: two NumpyPathDataset objects
        """
        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        dataset1.scratch_files = self.scratch_files[0:index]
        dataset2.scratch_files = self.scratch_files[index:]
    
        dataset1.npy_files = self.npy_files[0:index]
        dataset2.npy_files = self.npy_files[index:]

        dataset1._init_samplebuffer()
        dataset2._init_samplebuffer()

        return dataset1, dataset2
  
    def _load_batch_from_filelist(self, batch_paths):
        """Takes a list of numpy files, loads the numpy files, stacks them, and inserts an extra color channel"""

        batch = [np.load(path).astype('float32') for path in batch_paths]
        if len(batch) > 0:
            batch = np.stack(batch)
            batch = batch[:, np.newaxis, ...]

        return batch

    def batch(self, batch_size, auto_repeat = True, verbose=False):
        """Returns a batch of numpy arrays from the sample buffer.
        Parameters:
            batch_size: size of the batch that should be returned
            auto_repeat: automatically call NumpyPathDataset.repeat() to refill the sample buffer with the contents of self.scratch_files (in randomized order).
            verbose: will print the path names for the batches. Typically only for debugging.
        Returns:
            Batch of samples (np.array)
        """
        if batch_size > len(self.samplebuffer):
            if auto_repeat:
                self.repeat()
                # Call batch_new again, since in theory if batch_size >> len(self.scratch_files), the samplebuffer may need to be extended multiple times
                return self.batch(batch_size, auto_repeat, verbose)
            else:
                # Just return whatever is left in the samplebuffer. Note that this will be fewer samples than the specified batch_size and may cause problems in the code
                batch_paths = self.samplebuffer
        else:
            # First part of the samplebuffer becomes the batch, the rest becomes the new samplebuffer
            batch_paths = self.samplebuffer[0:batch_size]
            self.samplebuffer = self.samplebuffer[batch_size:]
        
        if verbose:
            print("Got batch:")
            for element in batch_paths:
                print(element)

        return self._load_batch_from_filelist(batch_paths)

    def batch_mpi(self, batch_size, auto_repeat = True, verbose=False):
        """Returns a batch of numpy arrays from the sample buffer of rank 0.
        If all workers sample individually (with batch()), some samples may be seen multiple times, while others may not be seen by any worker.
        For batch_mpi, since all workers sample from the buffer of rank 0 rather than sampling individually, all samples will be seen by a single worker before the dataset is repeated.
        From a parallel computing sense, batch() duplicates work, whil batch_mpi() distributes work.
        Note that all workers will have to call batch_mpi in order to avoid a deadlock or other indeterminate behaviour.
        Parameters:
            batch_size: size of the batch that should be returned (per worker)
            auto_repeated: have rank 0 automatically call NumpyPathDataset.repeat() to refill the sample buffer with the contents of self.scratch_files (in randomized order).
            verbose: will print the path names for the batches. Typically only for debugging
        Returns:
            Returns:
            Batch of samples (np.array)
        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            global_batch_size = batch_size * MPI.COMM_WORLD.Get_size()
            if global_batch_size > len(self.samplebuffer):
                if auto_repeat:
                    self.repeat()
                    # Call batch_new again, since in theory if batch_size >> len(self.scratch_files), the samplebuffer may need to be extended multiple times
                    return self.batch_mpi(batch_size, auto_repeat, verbose)
                else:
                    # Just return whatever is left in the samplebuffer. Note that this will be fewer samples than the specified batch_size and may cause problems in the code
                    batch_paths = self.samplebuffer
                    # Pad with 'None', otherwise our reshape will fail later on
                    while len(batch_paths) % MPI.COMM_WORLD.Get_size() > 0:
                        batch_paths.append(None)
            else:
                # First part of the samplebuffer becomes the batch, the rest becomes the new samplebuffer
                batch_paths = self.samplebuffer[0:global_batch_size]
                self.samplebuffer = self.samplebuffer[global_batch_size:]
            # Reshape into an array of which the first dim is MPI.COMM_WORLD.Get_size(). This is the dim along which scatter scatters the lists over workers.
            batch_paths = np.transpose(np.array(batch_paths).reshape(-1, MPI.COMM_WORLD.Get_size())).tolist()
        else: # initialize batch_paths for all other ranks
            batch_paths = None

        # Now that rank 0 has read the batch_paths, call MPI scatter
        if verbose:
            print(f"Worker: {MPI.COMM_WORLD.Get_rank()}. Before scatter batch_paths: {batch_paths}")

        batch_paths = MPI.COMM_WORLD.scatter(batch_paths, root=0)

        # Strip 'None' from the end of batch_paths
        while len(batch_paths) > 0 and batch_paths[-1] is None:
            batch_paths.pop()

        if verbose:
            print(f"Worker: {MPI.COMM_WORLD.Get_rank()}. Got batch: {batch_paths}")

        return self._load_batch_from_filelist(batch_paths)
        
    def repeat(self):
        """Repeat the dataset. Will be called internally once the dataset runs out of samples if auto_repeat is set."""
        # Note: the [:] on self.scratch_files is needed to make sure the list gets duplicated - otherwise self.scratch_files will also get shuffled
        new_samplebuffer = self.scratch_files[:]
        random.shuffle(new_samplebuffer)
        self.samplebuffer.extend(new_samplebuffer)

    def print_samplebuffer(self):
        for path in self.samplebuffer:
            print(path)


## This file can be tested by calling it directly
if __name__ == "__main__":

    import os
    import numpy as np

    a=np.zeros([5, 16, 16])
    
    scratch = os.path.join(os.getenv('TMPDIR'), os.getenv('USER'))
    savepath = os.path.join(scratch, 'datadir/')

    os.makedirs(savepath, exist_ok = True)

    # Create 10 dummy files
    for i in range(10):
        filename = os.path.join(savepath, str(i).zfill(3) + '.npy')
        np.save(filename, a)

    print(f"Savepath: {savepath}")

    if MPI.COMM_WORLD.Get_size() > 1:
        print(f"\nWorker: {MPI.COMM_WORLD.Get_rank()}. Testing batch_mpi(), no autorepeat")
        npy_data = NumpyPathDataset(savepath, scratch, True, True)
        npy_data.batch_mpi(3, False, True)
        npy_data.batch_mpi(3, False, True)

        print(f"\nWorker: {MPI.COMM_WORLD.Get_rank()}. Testing batch_mpi(), with autorepeat")
        npy_data = NumpyPathDataset(savepath, scratch, True, True)
        npy_data.batch_mpi(3, True, True)
        npy_data.batch_mpi(3, True, True)
    else:
        print(f"\nTesting batch(), no autorepeat")
        npy_data = NumpyPathDataset(savepath, scratch, True, True)
        npy_data.batch(7, False, True)
        npy_data.batch(7, False, True)

        print(f"\nTesting batch(), with autorepeat")
        npy_data = NumpyPathDataset(savepath, scratch, True, True)
        npy_data.batch(7, True, True)
        npy_data.batch(7, True, True)

    
