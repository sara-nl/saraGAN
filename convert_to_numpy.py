import tensorflow as tf
import numpy as np
import os
from functools import partial

from tensorflow.data.experimental import AUTOTUNE

def _parse_serialized(example_proto, shape):
    keys_to_features = {'image': tf.io.FixedLenFeature((int(np.product(shape)),), dtype=tf.float32)}
    parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
    return parsed_features['image']


def _normalize_to_rng(tensor, rng=(-1, 1)):
    tensor_min = tf.reduce_min(tensor)
    tensor_max = tf.reduce_max(tensor)
    return (rng[1] - rng[0]) * (tensor - tensor_min) / (tensor_max - tensor_min) - rng[1]


def load_lidc_idri_dataset_from_tfrecords(path, batch_size, shape, horovod=False):
    
    print(f"Dataset output shape: {shape}")

    filenames = os.listdir(path)
    abs_filenames = [os.path.join(path, filename) for filename in filenames]
    assert all(filename.endswith('.tfrecord') for filename in abs_filenames)
    
    dataset = tf.data.TFRecordDataset(abs_filenames)
    
    dataset = dataset.map(partial(_parse_serialized, shape=shape), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(partial(tf.reshape, shape=shape))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset

root = '/project/davidr/lidc_idri/tfrecords/tfrecords_lanczos/'
output = '/project/davidr/lidc_idri/npys/lanczos/'

if not os.path.exists(output):
    os.makedirs(output)

for size in (512,):
    folder = os.path.join(root, f'tfrecords_{size}x{size}x{size}')
    output_folder = os.path.join(output, f'{size}x{size}')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    shape_z = size // 4
    shape = (shape_z, size, size)
    dataset = load_lidc_idri_dataset_from_tfrecords(folder, 1, shape)
    
    
    for i, tensor in enumerate(dataset):
        array = tensor.numpy()
        
        
        filename = os.path.join(output_folder, f'{i:04}.npy')
        np.save(filename, array)
    
    
    
    