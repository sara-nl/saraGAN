import tensorflow as tf


def load_data(dataset):
    # (mnist_images, mnist_labels), _ = \
    # tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

    if dataset == 'cifar10':
        (images, labels), _ = \
            tf.keras.datasets.cifar10.load_data()
        n_classes = 10
        epoch_size = 50000

    elif dataset == 'mnist':
        (images, labels), _ = \
            tf.keras.datasets.mnist.load_data()
        n_classes = 10
        epoch_size = 50000

    elif dataset == 'fmnist':
        (images, labels), _ = tf.keras.datasets.fashion_mnist.load_data()
        n_classes = 10
        epoch_size= 50000

    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

    if images.ndim == 3:
        images = images[..., tf.newaxis]

    if labels.ndim == 2:
        labels = labels.squeeze(-1)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast((images - 127.5) / 127.5, tf.float32),
         tf.cast(labels, tf.int64))
    )

    def _parse_function(image, label):
        image_resized = tf.image.resize(image, (32, 32))
        return image_resized, label

    if images.shape[1] < 32:
        dataset = dataset.map(_parse_function)

    dataset = dataset.repeat().shuffle(10000).batch(128)
    return epoch_size, n_classes, dataset
