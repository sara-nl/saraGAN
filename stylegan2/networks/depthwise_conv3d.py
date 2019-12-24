import tensorflow as tf
import time

def group_conv3d(x, filter, groups):
    inputs = tf.split(x, groups, axis=1)
    filters = tf.split(filter, groups, axis=-2)
    output = tf.concat(
        [tf.nn.conv3d(i, f,
                      strides=[1, 1, 1, 1, 1],
                      padding='SAME',
                      data_format='NCDHW')
         for i, f in zip(inputs, filters)], axis=1)

    return output


if __name__ == '__main__':
    x = tf.random.normal(shape=(2, 8, 8, 32, 32))
    filter = tf.random.normal(shape=(3, 3, 3, 8, 2))
    out = group_conv3d(x, filter, groups=x.shape[1])

    with tf.Session() as sess:

        start = time.time()
        for _ in range(1000):
            sess.run(out)
        end = time.time()

        print(f"duration: {end - start}")

