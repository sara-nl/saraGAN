from networks.ops import get_weight, conv3d
import tensorflow as tf
import os
import time
from tqdm import tqdm
import numpy as np
import psutil


os.environ['OMP_NUM_THREADS'] = str(16)
n = 10


def conv2d(x, fmaps, kernel, activation, param=None):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], activation, param=param)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')


def dw_conv3d(x, fmaps, kernel, activation, param=None):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], activation, param=param)
    w = tf.cast(w, x.dtype)
    return tf.nn.depthwise_conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')


def inverted_residual(x, c_out, expand_ratio=2):

    shape = x.get_shape().as_list()
    c_in = shape[1]
    hidden_dim = int(round(c_in * expand_ratio))
    with tf.variable_scope('excite'):
        x = conv2d(x, hidden_dim, kernel=1, activation='leaky_relu', param=0.2)
    with tf.variable_scope('dw_conv'):
        x = dw_conv3d(x, hidden_dim, 3, 'linear')
    with tf.variable_scope('squeeze'):
        x = conv2d(x, c_out, 1, 'leaky_relu', param=0.2)

    return x


x = tf.random.normal((4, 32, 128, 128))
with tf.variable_scope('b1'):
    y = inverted_residual(x, 32)
with tf.variable_scope('b2'):
    z = inverted_residual(x, 32)

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

x = tf.random.normal((4, 32, 128, 128))
with tf.variable_scope('b1'):
    y = conv2d(x, 512, 3, 'leaky_relu', param=0.2)
with tf.variable_scope('b2'):
    z = conv2d(x, 512, 3, 'leaky_relu', param=0.2)

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




