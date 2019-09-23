import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter
from distutils.dir_util import copy_tree
import os
import tensorflow as tf
from functools import partial
import numpy as np
import nvgpu
import psutil
import horovod.tensorflow as hvd
import math
from metrics import sliced_wasserstein_distance_3d


def print_gpu_info():
    gpu_info = nvgpu.gpu_info()
    
    for d in gpu_info:
    #     print(f"\n{d['type']} \t Memory Used: {d['mem_used']} \t Total Memory {d['mem_total']}\n")
        yield d['type'], d['mem_used']
        
        
def print_cpu_info():
    process = psutil.Process(os.getpid())
    return os.getpid(), process.memory_info().rss / (1024 ** 2)
    
    
AUTOTUNE = tf.data.experimental.AUTOTUNE


def save_array_as_gif(file: str, slice_array):
    
    fig = plt.figure()
    
    plots = []
    
    fps = slice_array.shape[0] / 4 
    
    for i in range(slice_array.shape[0]):
        plot = plt.imshow(slice_array[i, :, :], cmap='gray')
        plots.append([plot])
    
    anim = ArtistAnimation(fig, plots)
    
    # writer = PillowWriter(fps=fps)
    anim.save(file, writer='imagemagick')
    plt.close()
    

def generate_gif(fakes, originals, output_dir, epoch):
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
                
    fake_grid = transform_batch_to_image_grid(fakes)
    original_grid = transform_batch_to_image_grid(originals)
    
    fake_grid = np.clip(fake_grid, -1, 1)
    original_grid = np.clip(original_grid, -1, 1)
    
    padding = np.ones((original_grid.shape[0], 4, *original_grid.shape[2:]))
    image = np.concatenate([fake_grid, padding, original_grid], axis=1)
    size = originals.shape[1]
    output_path = os.path.join(output_dir, f'{size}x{size}x{size}_{epoch:04}.gif')
    save_array_as_gif(output_path, image)
    
    
def transform_batch_to_image_grid(array):
    
    dim = int(array.shape[0] ** 0.5)
    assert dim ** 2 == array.shape[0], "Invalid shape for grid, first dimension has no perfect square."
    
    r, c, = dim, dim
    array_width = array.shape[2]
    array_height = array.shape[3]

    grid = np.empty((array.shape[1], r * array_width, c * array_height))
        
    for i in range(r):
        for j in range(c):
            x_loc = i * array_width
            y_loc = j * array_height

            image = array[i * r + j]
            grid[:, x_loc: x_loc + array_width, y_loc: y_loc + array_height] = image
    
    return grid    

    
def _parse_serialized(example_proto, shape):
    keys_to_features = {'image': tf.io.FixedLenFeature((int(np.product(shape)),), dtype=tf.float32)}
    parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
    return parsed_features['image']


def _normalize_to_rng(tensor, rng=(-1, 1)):
    tensor_min = tf.reduce_min(tensor)
    tensor_max = tf.reduce_max(tensor)
    return (rng[1] - rng[0]) * (tensor - tensor_min) / (tensor_max - tensor_min) - rng[1]


def load_lidc_idri_dataset_from_tfrecords(path, batch_size, shape, horovod=False):
    
    if horovod:
        seed_adjustment = hvd.rank()
        tf.set_random_seed(42 + seed_adjustment)
        np.random.seed(42 + seed_adjustment)

    filenames = os.listdir(path)
    abs_filenames = [os.path.join(path, filename) for filename in filenames]
    assert all(filename.endswith('.tfrecord') for filename in abs_filenames)
    
    dataset = tf.data.TFRecordDataset(abs_filenames)
    
    dataset = dataset.map(partial(_parse_serialized, shape=shape), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(_normalize_to_rng, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(partial(tf.reshape, shape=shape))
    
    dataset = dataset.shuffle(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset
    
    
def nearest_square_root(n):
    return int(math.pow(math.floor(math.sqrt(n)), 2))    
    
    
def get_wasserstein_batch(dataset, min_size=64):
    
    return_batch = None  # Placeholder
    
    for batch in dataset.take(min_size):
        if return_batch is None:
            return_batch = batch
        
        else:
            return_batch = np.concatenate([return_batch, batch], axis=0)
        
        if len(return_batch) >= min_size:
            return return_batch
        
      
def write_wasserstein_distances(real_batch, fake_batch, step):
    assert real_batch.shape == fake_batch.shape
    min_res = 8
    swd = sliced_wasserstein_distance_3d(real_batch, fake_batch, resolution_min=min_res)
    
    for i in reversed(range(len((swd)))):
        lod = min_res * 2 ** i
        tf.summary.scalar(f"swd_real_real_{lod}", swd[i][0], step=step)
        tf.summary.scalar(f"swd_fake_real_{lod}", swd[i][1], step=step)
        