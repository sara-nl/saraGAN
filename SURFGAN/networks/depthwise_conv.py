import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time

batch = 64
in_c = 64
out_c = 64
s = 64
k = 3
n = 100

x_in = np.random.randn(batch, in_c, s, s)
w_in = np.random.randn(k, k, in_c, out_c)

# x = tf.constant(x_in, dtype=np.float32)
x = tf.random.normal(shape=x_in.shape)
w = tf.Variable(w_in, dtype=np.float32)
y = tf.nn.depthwise_conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
loss = tf.reduce_sum(y)
optim = tf.train.GradientDescentOptimizer(1e-5)
grad = optim.compute_gradients(loss, var_list=[w])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start = time.time()
    for _ in tqdm(range(n)):
        out = sess.run(grad)
    end = time.time()
    print(start - end)
sess.close()
