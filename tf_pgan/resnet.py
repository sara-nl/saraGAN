import tensorflow as tf
import numpy as np


def resnet_block(x):
    x_shape = x.get_shape().as_list()
    channels = min(512, x_shape[1] * 4)
    with tf.variable_scope('conv1'):
        x_1 = tf.layers.conv3d(x, channels, 3, padding='same', activation='relu',
                               data_format='channels_first')

    with tf.variable_scope('conv2'):
        x_2 = tf.layers.conv3d(x_1, channels, 3, padding='same', activation='relu',
                               data_format='channels_first')

    x = x_2 + 0.3 * x_1

    if x_shape[-3] > 1:
        x = tf.layers.max_pooling3d(x, (2, 2, 2), (2, 2, 2), data_format='channels_first')
    return x


def resnet(x: tf.Variable, is_reuse: bool = False) -> tf.Variable:
    """input shape: [bs, chan, dimz, dimy, dimx]
    first phase: [bs, chan, 1, 4, 4]
    """

    x_shape = x.get_shape().as_list()
    num_blocks = int(np.log2(x_shape[-1]) - 1)

    with tf.variable_scope('resnet') as scope:

        if is_reuse:
            scope.reuse_variables()

        for i in range(num_blocks):
            with tf.variable_scope(f'resnet_block_{i}'):
                print(x.shape)
                x = resnet_block(x)

        x = tf.reshape(x, [tf.shape(x)[0], -1])
        with tf.variable_scope('classifier'):
            x = tf.layers.dense(x, 1)

    return x


def test():

    for phase in range(1, 8):
        tf.reset_default_graph()
        spatial_shape = list(np.array([1, 4, 4]) * 2 ** (phase - 1))
        x = tf.random.normal([1, 1] + spatial_shape)
        output = resnet(x)
        print(output.shape)


if __name__ == '__main__':
    test()
