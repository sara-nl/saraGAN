import tensorflow as tf
import os
from functools import partial
import numpy as np
from distutils.dir_util import copy_tree

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_and_preprocess_image_from_path(path, channels, reshape_size=None):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    if reshape_size:
        image = tf.image.resize(image, reshape_size)
    return image


def load_and_preprocess_numpy_image_label(images, labels, reshape_size=None):
    if images.ndim == 3:
        images = images[..., tf.newaxis]

    if labels.ndim == 2:
        labels = labels.squeeze(-1)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast((images - 127.5) / 127.5, tf.float32),
         tf.cast(labels, tf.int64))
    )

    def _parse_function(image, label, size):
        image_resized = tf.image.resize(image, size)
        return image_resized, label

    if reshape_size:
        dataset = dataset.map(partial(_parse_function, size=reshape_size))

    return dataset


def load_data(dataset, batch_size):

    if dataset == 'cifar10':
        (images, labels), _ = \
            tf.keras.datasets.cifar10.load_data()
        image_ds = load_and_preprocess_numpy_image_label(images, labels)
        n_classes = len(np.unique(labels))
        epoch_size = len(images)

    elif dataset == 'mnist':
        (images, labels), _ = \
            tf.keras.datasets.mnist.load_data()
        image_ds = load_and_preprocess_numpy_image_label(images, labels, reshape_size=(32, 32))
        n_classes = len(np.unique(labels))
        epoch_size = len(images)

    elif dataset == 'fmnist':
        (images, labels), _ = tf.keras.datasets.fashion_mnist.load_data()
        image_ds = load_and_preprocess_numpy_image_label(images, labels, reshape_size=(32, 32))
        n_classes = len(np.unique(labels))
        epoch_size = len(images)

    elif dataset == 'celeba':
        dataset_path = os.path.join('/lustre4', '2', 'managed_datasets', 'CelebA', 'img_align_celeba')
        scratch_path = os.path.join('/scratch-shared', dataset)

        if not os.path.exists(scratch_path):
            print("Moving files to scratch space...")
            copy_tree(dataset_path, scratch_path)
            print("Done!")

        all_image_paths = os.listdir(scratch_path)
        all_image_paths = [os.path.join(scratch_path, path) for path in all_image_paths]
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(partial(load_and_preprocess_image_from_path, channels=3,
                                       reshape_size=(64, 64)), num_parallel_calls=AUTOTUNE)
        epoch_size = len(all_image_paths)
        n_classes = None

    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

    image_ds = image_ds.repeat().shuffle(epoch_size).batch(batch_size)
    return epoch_size, n_classes, image_ds
