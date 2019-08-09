import tensorflow as tf


def load_data(dataset):
    # (mnist_images, mnist_labels), _ = \
    # tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

    if dataset == 'cifar10':
        (images, labels), _ = \
            tf.keras.datasets.cifar10.load_data()

    elif dataset == 'mnist':
        (images, labels), _ = \
            tf.keras.datasets.mnist.load_data()

    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

    if images.ndim == 3:
        images = images[..., tf.newaxis]

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(images / 255.0, tf.float32),
         tf.cast(labels, tf.int64))
    )

    def _parse_function(image, label):
        image_resized = tf.image.resize(image, (32, 32))
        return image_resized, label

    if images.shape[1] < 32:
        dataset = dataset.map(_parse_function)

    dataset = dataset.repeat().shuffle(10000).batch(128)
    return dataset
