from networks.ops import get_weight, conv3d
import tensorflow as tf
import os
import time
from tqdm import tqdm
import numpy as np
import psutil


os.environ['MPI_NUM_THREADS'] = str(16)
n = 10


def depthwise_3d(x, fmaps, kernel, groups, activation, param=None):

    bs, in_c, d, h, w = x.get_shape().as_list()
    assert groups == in_c  # Only support depthwise now.
    fmaps = fmaps // groups
    filter = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], activation, param=param)

    def f(_, inputs):
        x = tf.nn.conv3d(inputs[0], inputs[1], strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')
        return x

    x_scan = tf.transpose(x, (1, 0, 2, 3, 4))[:, :, tf.newaxis, ...]
    filter_scan = tf.transpose(filter, (3, 0, 1, 2, 4))[..., tf.newaxis, :]
    output_shape = [bs, fmaps, d, w, h]
    c = tf.scan(f, (x_scan, filter_scan), initializer=tf.zeros(output_shape), parallel_iterations=int(os.environ['OMP_NUM_THREADS']))
    c = tf.transpose(tf.squeeze(c, axis=2), [1, 0, 2, 3, 4])
    return c


def inverted_residual(x, c_out, expand_ratio=6):

    shape = x.get_shape().as_list()
    c_in = shape[1]
    hidden_dim = int(round(c_in * expand_ratio))
    with tf.variable_scope('excite'):
        x = conv3d(x, hidden_dim, kernel=1, activation='leaky_relu', param=0.2)
    with tf.variable_scope('dw_conv'):
        x = depthwise_3d(x, hidden_dim, kernel=3, groups=hidden_dim, activation='leaky_relu', param=0.2)

    with tf.variable_scope('squeeze'):
        x = conv3d(x, c_out, 1, 'leaky_relu', param=0.2)

    return x


x = tf.random.normal((4, 256, 8, 32, 32))
with tf.variable_scope('b1'):
    y = inverted_residual(x, 256)
with tf.variable_scope('b2'):
    z = inverted_residual(x, 256)

loss = tf.reduce_sum(z)
optim = tf.train.GradientDescentOptimizer(1e-5)
train = optim.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(n)):
        if i == 5:
            start = time.time()
            process = psutil.Process(os.getpid())
            print(f'Mobile Block Memory: {process.memory_percent():.2f}%')  # in bytes
        sess.run(train)

    end = time.time()

    print(f"Mobile block: {end - start}")
    parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print(f"Mobile block Parameters: {parameters}")

sess.close()
tf.reset_default_graph()
time.sleep(5)

x = tf.random.normal((4, 256, 8, 32, 32))
with tf.variable_scope('b1'):
    y = conv3d(x, 256, 3, 'leaky_relu', param=0.2)
with tf.variable_scope('b2'):
    z = conv3d(x, 256, 3, 'leaky_relu', param=0.2)

loss = tf.reduce_sum(z)
optim = tf.train.GradientDescentOptimizer(1e-5)
train = optim.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(n)):
        if i == 5:
            start = time.time()
            process = psutil.Process(os.getpid())
            print(f'Conv Block Memory: {process.memory_percent():.2f}%')  # in bytes
        sess.run(train)

    end = time.time()

    print(f"Conv block: {end - start}")
    parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print(f"Conv block Parameters: {parameters}")




