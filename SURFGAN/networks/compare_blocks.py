import tensorflow as tf
from networks.ops import conv3d, apply_bias, act
import time
import psutil
from tqdm import tqdm
import numpy as np
import os

os.environ['OMP_NUM_THREADS'] = str(16)

B = 1
D, H, W = 32, 128, 128
C_IN = 64
C_OUT = 128
SHAPE = (B, C_IN, D, H, W)
N = 10


def discriminator_block(x, filters_in, filters_out, activation, param=None):
    with tf.variable_scope('conv_1'):
        x = conv3d(x, filters_in, 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)

    with tf.variable_scope('conv_2'):
        x = conv3d(x, filters_out, 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    return x


def run_basic():
    tf.reset_default_graph()
    x = tf.random.normal(SHAPE)
    y = discriminator_block(x, C_IN, C_OUT, 'leaky_relu', 0.2)
    loss = tf.reduce_sum(y)
    optim = tf.train.GradientDescentOptimizer(1e-5)
    train = optim.minimize(loss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(N)):
            if i == 5:
                start = time.time()
                process = psutil.Process(os.getpid())
                mem = process.memory_percent()

            sess.run(train)

        end = time.time()

        print(f'Basic Block Memory: {mem:.2f}%')  # in bytes
        print(f"Basic block: {end - start}")
        parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(f"Basic block Parameters: {parameters}")
    sess.close()
    time.sleep(5)

# ======================================================================================================================
# ======================================================================================================================


def d_block_deep(x, filters_in, filters_out, activation, param=None):
    with tf.variable_scope('conv_1'):
        x = conv3d(x, int(filters_in / np.sqrt(2)), 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    with tf.variable_scope('conv_2'):
        x = conv3d(x, int(filters_in / np.sqrt(2)), 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    with tf.variable_scope('conv_3'):
        x = conv3d(x, filters_out, 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)

    return x


def run_deep():
    tf.reset_default_graph()
    x = tf.random.normal(SHAPE)
    y = d_block_deep(x, C_IN, C_OUT, 'leaky_relu', 0.2)
    loss = tf.reduce_sum(y)
    optim = tf.train.GradientDescentOptimizer(1e-5)
    train = optim.minimize(loss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(N)):
            if i == 5:
                start = time.time()
                process = psutil.Process(os.getpid())
                mem = process.memory_percent()

            sess.run(train)

        end = time.time()

        print(f'Deep Block Memory: {mem:.2f}%')  # in bytes
        print(f"Deep block: {end - start}")
        parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(f"Deep block Parameters: {parameters}")
    sess.close()
    time.sleep(5)

# ======================================================================================================================
# ======================================================================================================================



def wide_block(x, filters_in, filters_out, activation, param=None):
    with tf.variable_scope('conv_1'):
        x = conv3d(x, int(filters_out * 1.3), 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    return x


def run_wide():
    tf.reset_default_graph()
    x = tf.random.normal(SHAPE)
    y = wide_block(x, C_IN, C_OUT, 'leaky_relu', 0.2)
    loss = tf.reduce_sum(y)
    optim = tf.train.GradientDescentOptimizer(1e-5)
    train = optim.minimize(loss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(N)):
            if i == 5:
                start = time.time()
                process = psutil.Process(os.getpid())
                mem = process.memory_percent()
            sess.run(train)

        end = time.time()

        print(f'Wide Block Memory: {mem:.2f}%')  # in bytes
        print(f"Wide block: {end - start}")
        parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(f"Wide block Parameters: {parameters}")
    sess.close()
    time.sleep(5)


# ======================================================================================================================
# ======================================================================================================================


def bottleneck_block(x, filters_in, filters_out, activation, param=None):
    with tf.variable_scope('conv_1'):
        x = conv3d(x, int(filters_out / 2), 1, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)

    with tf.variable_scope('conv_2'):
        x = conv3d(x, int(filters_out / 2), 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)

    with tf.variable_scope('conv_3'):
        x = conv3d(x, int(filters_out), 1, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    return x


def run_bottleneck():
    tf.reset_default_graph()
    x = tf.random.normal(SHAPE)
    y = bottleneck_block(x, C_IN, C_OUT, 'leaky_relu', 0.2)
    loss = tf.reduce_sum(y)
    optim = tf.train.GradientDescentOptimizer(1e-5)
    train = optim.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(N)):
            if i == 5:
                start = time.time()
                process = psutil.Process(os.getpid())
                mem = process.memory_percent()
            sess.run(train)

        end = time.time()

        print(f'Bottleneck Block Memory: {mem:.2f}%')  # in bytes
        print(f"Bottleneck block: {end - start}")
        parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(f"Bottleneck block Parameters: {parameters}")

    sess.close()
    time.sleep(5)


# ======================================================================================================================
# ======================================================================================================================


def fast_block(x, filters_in, filters_out, activation, param=None):
    with tf.variable_scope('conv_1'):
        x = conv3d(x, int(filters_out / np.sqrt(3)), 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)

    with tf.variable_scope('conv_2'):
        x = conv3d(x, int(filters_out / np.sqrt(3)), 1, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)

    with tf.variable_scope('conv_3'):
        x = conv3d(x, int(filters_out / np.sqrt(3)), 3, activation, param=param)
        x = apply_bias(x)
        x = act(x, activation, param=param)
    return x


def run_fast():
    tf.reset_default_graph()
    x = tf.random.normal(SHAPE)
    y = fast_block(x, C_IN, C_OUT, 'leaky_relu', 0.2)
    loss = tf.reduce_sum(y)
    optim = tf.train.GradientDescentOptimizer(1e-5)
    train = optim.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(N)):
            if i == 5:
                start = time.time()
                process = psutil.Process(os.getpid())
                mem = process.memory_percent()
            sess.run(train)

        end = time.time()

        print(f"Fast block: {end - start}")
        print(f'Fast Block Memory: {mem:.2f}%')  # in bytes
        parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(f"Fast block Parameters: {parameters}")

    sess.close()
    time.sleep(5)


if __name__ == '__main__':
    run_wide()
    # run_bottleneck()
    # run_fast()
    run_basic()
    # run_deep()

