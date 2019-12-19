import numpy as np
from tqdm import tqdm
import time
import os

os.environ['OMP_NUM_THREADS'] = str(16)
import tensorflow as tf

batch = 4
in_c = 16
out_c = 32
s = 32
k = 3
n = 10

x_in = np.random.randn(batch, in_c, s, s, s)
w_in = np.random.randn(k, k, k, in_c, out_c // in_c)

x = tf.random.normal(x_in.shape)
w = tf.Variable(w_in, dtype=tf.float32)

def scan_depthwise_3d(x, w):

    def f(_, inputs):
        x = tf.nn.conv3d(inputs[0], inputs[1], strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')
        return x

    x_scan = tf.transpose(x, (1, 0, 2, 3, 4))[:, :, tf.newaxis, ...]
    w_scan = tf.transpose(w, (3, 0, 1, 2, 4))[..., tf.newaxis, :]
    output_shape = [batch, out_c // in_c, s, s, s]
    c = tf.scan(f, (x_scan, w_scan), initializer=tf.zeros(output_shape), parallel_iterations=int(os.environ['OMP_NUM_THREADS']))
    c = tf.reshape(tf.transpose(c, [1, 0, 2, 3, 4, 5]), (batch, out_c, s, s, s))
    return c

z = scan_depthwise_3d(x, w)
print(z.shape)
loss = tf.reduce_sum(z)
optim = tf.train.GradientDescentOptimizer(1e-5)
train = optim.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(n)):
        if i == 5:
            start = time.time()
        out = sess.run(train)
    end = time.time()
    print(end - start)
sess.close()
time.sleep(5)

tf.reset_default_graph()
# x = tf.constant(x_in, dtype=np.float32)
x = tf.random.normal(x_in.shape)
w_in = np.random.randn(k, k, k, in_c, out_c)
w = tf.Variable(w_in, dtype=np.float32)
a = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')
print(a.shape)
loss = tf.reduce_sum(a)
optim = tf.train.GradientDescentOptimizer(1e-5)
train = optim.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(n)):
        if i == 5:
            start = time.time()
        out = sess.run(train)
    end = time.time()
    print(end - start)
