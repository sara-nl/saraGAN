import time

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import ast
from multiprocessing import Pool
import os
import horovod.tensorflow as hvd

from dataset import NumpyPathDataset

def scale_lr(g_lr, d_lr, g_scaling, d_scaling, horovod):
    """Scales the learning rates if horovod is used.
    Parameters:
        g_lr: generator learning rate
        d_lr: discriminator learning rate
        g_scaling: scaling method to use for g_lr
        d_scaling: scaling method to use for d_lr
        horovod: if horovod is enabled (bool)
    Returns:
        g_lr, d_lr (scaled for horovod parallelism, if applicable)
    """
    if horovod:
        if g_scaling == 'sqrt':
            g_lr = g_lr * np.sqrt(hvd.size())
        elif g_scaling == 'linear':
            g_lr = g_lr * hvd.size()
        elif g_scaling == 'none':
            pass
        else:
            raise ValueError(g_scaling)

        if d_scaling == 'sqrt':
            d_lr = d_lr * np.sqrt(hvd.size())
        elif d_scaling == 'linear':
            d_lr = d_lr * hvd.size()
        elif d_scaling == 'none':
            pass
        else:
            raise ValueError(d_scaling)

    return g_lr, d_lr
            
def get_num_metric_samples(num_metric_samples, batch_size, global_size):
    """Returns the number of samples to be trained on before metrics are recalculated"""
    if not num_metric_samples:
        if batch_size > 1:
            num_metric_samples = batch_size * global_size
        else:
            num_metric_samples = 2 * global_size
    else:
        num_metric_samples = num_metric_samples
    return num_metric_samples

# TODO: I could probably move the normalization to this function, and do it based on arguments...
def get_image_input_tensor(phase, batch_size, start_shape, noise_stddev=None):
    """Gets the input tensor for the network, initialized with the correct shape. Normally distributed noise can be added to make training more robust to noise patterns in the input.
    Parameters:
      phase: the current phase of the training
      batch_size: the current batch size (local batch size in case of horovd based training)
      start_shape: the starting shape of the lowest resolution images
      noise_stddev: standard deviation of the normal distribution that the noise is sampled from. Should probably be of the same order of the noise in your input images (after normalization, if normalization is applied to the inputs).
    Returns:
      tf.Tensor that can be used as input tensor for the gan
    """
    start_shape = parse_tuple(start_shape)
    current_shape = [batch_size, get_num_channels(start_shape), *[size * 2 ** (phase - 1) for size in
                                                       get_base_shape(start_shape)[1:]]]
    real_image_input = tf.placeholder(shape=current_shape, dtype=tf.float32)
    real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * noise_stddev
    return real_image_input

def get_xy_dim(phase, start_shape):
    """Get the dimensions of the current images in xy"""
    start_shape = parse_tuple(start_shape)
    start_resolution = start_shape[-1]
    size = start_resolution * (2 ** (phase - 1))
    return size

def get_numpy_dataset(phase, starting_phase, start_shape, dataset_path, scratch_path, verbose):
    """Get the numpy dataset object"""
    size = get_xy_dim(phase, start_shape)

    data_path = os.path.join(dataset_path, f'{size}x{size}/')
    if verbose:
        print(f'Phase {phase}: reading data from dir {data_path}')
    npy_data = NumpyPathDataset(data_path, scratch_path, copy_files=(hvd.local_rank() == 0),
                                   is_correct_phase=phase >= starting_phase)
    return npy_data

def get_num_channels(start_shape):
    """Get the number of channels, based on the starting shape"""
    start_shape = parse_tuple(start_shape)
    return start_shape[0]

def get_num_phases(start_shape, final_shape):
    """Get the number of phases, derived from the start and final shapes"""
    start_shape = parse_tuple(start_shape)
    start_resolution = start_shape[-1]
    final_shape = parse_tuple(final_shape)
    final_resolution = final_shape[-1]
    return int(np.log2(final_resolution/start_resolution))

def get_base_shape(start_shape):
    """Get the base shape of the network, i.e. the shape of the first layer in the generator.
    Returns: A tuple representing the shape of the first layer at the base of the generator network"""
    start_shape = parse_tuple(start_shape)
    base_shape = (start_shape[0], start_shape[1], start_shape[2], start_shape[3])
    return base_shape

def get_filewriter(logdir, verbose):
    """Creates a tf.summary.FileWriter for logdir, but only if this is rank 0 (in case of MPI). Returns None for othre ranks"""
    if verbose:
        writer = tf.summary.FileWriter(logdir=logdir)
    else:
        writer = None
    return writer

def get_logdir(args):
    """Checks if a logdir was defined. If not, a logdir is created based on a timestamp"""
    if args.logdir is not None:
        logdir = args.logdir
    else:
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', args.architecture, timestamp)

    if get_verbosity(args.horovod, args.optuna_distributed):
        print(f"Saving files to {logdir}")
    return logdir

def get_verbosity(horovod, optuna_distributed):
    """Checks if this is an MPI process. If so, returns "True" only for rank 0
    Parameters:
    --------
      horovod: if this is a horovod enabled run (i.e. args.horovod = True). Boolean.
      optuna_distributed: if this is an optuna distributed run (i.e. args.optuna_distributed = True). Boolean
    Returns:
    --------
      Boolean indicating whether output should be printed (only for rank 0 if this is an MPI based run)
    """
    if horovod or optuna_distributed:
        verbose = hvd.rank() == 0
    else:
        verbose = True
    return verbose

def create_compute_metrics_dict(args):
    compute_metrics = {
        'compute_FID': args.compute_FID,
        'compute_swds': args.compute_swds,
        'compute_ssims': args.compute_ssims,
        'compute_psnrs': args.compute_psnrs,
        'compute_mses': args.compute_mses,
        'compute_nrmses': args.compute_nrmses,
    }
    
    return compute_metrics



# Op to update the learning rate according to a schedule
def lr_update(lr, intra_phase_step, steps_per_phase, lr_max, lr_increase, lr_decrease, lr_rise_niter, lr_decay_niter):
    """Update the learning rate according to a schedule.
    Args:
      lr: Tensor which contains the current learning rate that needs to be updated
      intra_phase_step: Step counter representing the number of images processed since the start of the current phase
      steps_per_phase: total number of steps in a phase
      lr_max: learning rate after increase (and before decrease) segments
      lr_increase: type of increase function to use (e.g. None, linear, exponential)
      lr_decrease: type of decrease function to use (e.g. None, linear, exponential)
      lr_rise_niter: number of iterations over which the increase from the minimum to the maximum value should happen
      lr_decay_niter: number of iterations over which the decrease from the maximum to the minumum value should happen.
    Returns: an Op that can be passed to session.run to update the learning (lr) Tensor
    """

    # Default starting point is that update_lr = lr_max. If there are no lr_increase or lr_decrease
    # functions specified, it stays like this.
    lr_update = lr_max

    # Is a learning rate schedule defined at all? (otherwise, immediately return a constant)
    if (lr_increase or lr_decrease):
        # Rather than if-else statements, the way to define a piecewiese function is through tf.cond

        # Prepare some variables:
        a = tf.cast(tf.math.divide(lr_max, 100), tf.float32)
        b_rise = tf.cast(tf.math.divide(np.log(100), lr_rise_niter), tf.float32)
        b_decay = tf.cast(tf.math.divide(np.log(100), lr_decay_niter), tf.float32)
        step_decay_start = tf.subtract(steps_per_phase, lr_decay_niter)
        remaining_steps = tf.subtract(steps_per_phase, intra_phase_step)

        # Define the different functions
        def update_increase_lin ():
            return tf.multiply(
                               tf.cast(tf.truediv(intra_phase_step, lr_rise_niter), tf.float32),
                               lr_max
                               )
        def update_increase_exp():
            return tf.multiply(
                                a,
                                tf.math.exp(tf.multiply(b_rise, tf.cast(intra_phase_step, tf.float32)))
                                )

        def update_decrease_lin():
            return tf.multiply(
                               tf.cast(tf.truediv(remaining_steps, lr_decay_niter), tf.float32),
                               lr_max
                               )

        def update_decrease_exp():
            return tf.multiply(
                                a,
                                tf.math.exp(tf.multiply(b_decay, tf.cast(remaining_steps, tf.float32)))
                                )

        def no_op():
            return lr_update

        if lr_increase == 'linear':
            # Are we in the increasing part? Return update_increase_lin function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step < lr_rise_niter, update_increase_lin, no_op)
        elif lr_increase == 'exponential':
            # Are we in the increasing part? Return update_increase_exp function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step < lr_rise_niter, update_increase_exp, no_op)
            
        if lr_decrease == 'linear':
            # Are we in the decreasing part? Return return update_decrease function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step > step_decay_start, update_decrease_lin, no_op) 
        elif lr_decrease == 'exponential':
            # Are we in the decreasing part? Return return update_decrease function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step > step_decay_start, update_decrease_exp, no_op) 
 
    return lr.assign(lr_update)

# Op to update the learning rate according to a schedule
def lr_update_numpy(lr, intra_phase_step, steps_per_phase, lr_max, lr_increase, lr_decrease, lr_rise_niter, lr_decay_niter):
    """Update the learning rate according to a schedule.
    Args:
      lr: Tensor which contains the current learning rate that needs to be updated
      intra_phase_step: Step counter representing the number of images processed since the start of the current phase
      steps_per_phase: total number of steps in a phase
      lr_max: learning rate after increase (and before decrease) segments
      lr_increase: type of increase function to use (e.g. None, linear, exponential)
      lr_decrease: type of decrease function to use (e.g. None, linear, exponential)
      lr_rise_niter: number of iterations over which the increase from the minimum to the maximum value should happen
      lr_decay_niter: number of iterations over which the decrease from the maximum to the minumum value should happen.
    Returns: an Op that can be passed to session.run to update the learning (lr) Tensor
    """
    # Is a learning rate schedule defined at all? (otherwise, immediately return a constant)
    if not (lr_increase or lr_decrease):
        return lr.assign(lr_max)
    else:
        # Are we in the increasing part?
        if intra_phase_step < lr_rise_niter:
            if not lr_increase:
                updated_lr = lr_max
            elif lr_increase == 'linear':
                updated_lr = (intra_phase_step / lr_rise_niter) * lr_max
            elif lr_increase == 'exponential':
                # Define lr at step 0 to be 1% of lr_max
                a = lr_max / 100
                # Make sure then when intra_phase_step = lr_rise_niter, the lr = lr_max
                b = np.log(100 / lr_rise_niter) 
                update_lr = a*np.exp(b*intra_phase_step)  
            else:
                raise NotImplementedError("Unsupported learning rate increase type: %s" % lr_increase)
        # Are we in the decreasing part?
        elif intra_phase_step > (steps_per_phase - lr_decay_niter):
            if not lr_decrease:
                updated_lr = lr_max
            if lr_decrease == 'linear':
                updated_lr = ((steps_per_phase - intra_phase_step) / lr_decay_niter) * lr_max
            elif lr_increase == 'exponential':
                # Define lr at the last step to be 1% of lr_max
                a = lr_max / 100
                # Make sure that when intra_phase_step == steps_per_phase - lr_decay_niter, lr_max is returned
                b = np.log(100 / lr_decay_niter)
                update_lr = a*np.exp(b*(steps_per_phase - intra_phase_step))
            else:
                raise NotImplementedError("Unsupported learning rate decrease type: %s" % lr_decrease)
        # Are we in the flat part?
        else:
            updated_lr = lr_max
            
        return lr.assign(updated_lr)


# log0 only logs from hvd.rank() == 0
def log0(string):
    if hvd.rank() == 0:
        print(string)

def parse_tuple(string):
    s = ast.literal_eval(str(string))
    return s


def count_parameters(scope):
    return sum(np.product(p.shape) for p in tf.trainable_variables(scope))


def image_grid(input_tensor, grid_shape, image_shape=(32, 32), num_channels=3):
    """Arrange a minibatch of images into a grid to form a single image.
    Args:
      input_tensor: Tensor. Minibatch of images to format, either 4D
          ([batch size, height, width, num_channels]) or flattened
          ([batch size, height * width * num_channels]).
      grid_shape: Sequence of int. The shape of the image grid,
          formatted as [grid_height, grid_width].
      image_shape: Sequence of int. The shape of a single image,
          formatted as [image_height, image_width].
      num_channels: int. The number of channels in an image.
    Returns:
      Tensor representing a single image in which the input images have been
      arranged into a grid.
    Raises:
      ValueError: The grid shape and minibatch size don't match, or the image
          shape and number of channels are incompatible with the input tensor.
    """
    if grid_shape[0] * grid_shape[1] != int(input_tensor.shape[0]):
        raise ValueError("Grid shape %s incompatible with minibatch size %i." %
                         (grid_shape, int(input_tensor.shape[0])))
    if len(input_tensor.shape) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.shape[1]) != num_features:
            raise ValueError("Image shape and number of channels incompatible with "
                             "input tensor.")
    elif len(input_tensor.shape) == 4:
        if (int(input_tensor.shape[1]) != image_shape[0] or
                int(input_tensor.shape[2]) != image_shape[1] or
                int(input_tensor.shape[3]) != num_channels):
            raise ValueError("Image shape and number of channels incompatible with "
                             "input tensor.")
    else:
        raise ValueError("Unrecognized input tensor format.")
    height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
    input_tensor = array_ops.reshape(
        input_tensor, tuple(grid_shape) + tuple(image_shape) + (num_channels,))
    input_tensor = array_ops.transpose(input_tensor, [0, 1, 3, 2, 4])
    input_tensor = array_ops.reshape(
        input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
    input_tensor = array_ops.transpose(input_tensor, [0, 2, 1, 3])
    input_tensor = array_ops.reshape(
        input_tensor, [1, height, width, num_channels])

    return input_tensor


def uniform_box_sampler(arr, min_width, max_width):
    """
    Extracts a sample cut from `arr`.

    Parameters:
    -----------
    arr : array
        The numpy array to sample a box from
    min_width : int or tuple
        The minimum width of the box along a given axis.
        If a tuple of integers is supplied, it my have the
        same length as the number of dimensions of `arr`
    max_width : int or tuple
        The maximum width of the box along a given axis.
        If a tuple of integers is supplied, it my have the
        same length as the number of dimensions of `arr`

    Returns:
    --------
    (slices, x) : A tuple of the slices used to cut the sample as well as
    the sampled subsection with the same dimensionality of arr.
        slice :: list of slice objects
        x :: array object with the same ndims as arr
    """
    if isinstance(min_width, (tuple, list)):
        assert len(min_width) == arr.ndim, 'Dimensions of `min_width` and `arr` must match'

    else:
        min_width = (min_width,) * arr.ndim
    if isinstance(max_width, (tuple, list)):
        assert len(max_width) == arr.ndim, 'Dimensions of `max_width` and `arr` must match'
    else:
        max_width = (max_width,) * arr.ndim

    slices = []
    for dim, mn, mx in zip(arr.shape, min_width, max_width):
        start = int(np.random.uniform(0, dim))
        stop = start + int(np.random.uniform(mn, mx + 1))
        slices.append(slice(start, stop))
    return slices, arr[slices]


class MPMap:
    def __init__(self, f):
        self.pool = Pool(int(os.environ['OMP_NUM_THREADS']))
        self.f = f

    def map(self, l: list):
        return self.pool.map_async(self.f, l)

    def close(self):
        self.pool.close()
