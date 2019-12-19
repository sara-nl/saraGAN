import tensorflow as tf
import time
from networks.ops import get_weight
import psutil
import tqdm
import os
import numpy as np

gopts = tf.GraphOptions(place_pruned_graph=True)
config = tf.ConfigProto(graph_options=gopts)
config.gpu_options.allow_growth = True

def group_conv3d(x, fmaps, kernel, groups, activation, param=None):
    assert kernel >= 1 and kernel % 2 == 1
    assert groups <= x.shape[1]
    assert fmaps >= groups

    fmaps = fmaps // groups

    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], activation, param=param)

    inputs = tf.split(x, groups, axis=1)
    filters = tf.split(w, groups, axis=-2)
    output = tf.concat(
        [tf.nn.conv3d(i, f,
                      strides=[1, 1, 1, 1, 1],
                      padding='SAME',
                      data_format='NCDHW')
         for i, f in zip(inputs, filters)], axis=1)

    return output


def conv3d(x, fmaps, kernel, activation, param=None):

    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], activation, param=param)
    w = tf.cast(w, x.dtype)
    assert kernel >= 1 and kernel % 2 == 1
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')

def tf_scan_depthwise_conv2d():
    def f( old,input):
        x_sample = input[0]
        kernel_sample = input[1]
        print(x_sample.shape, kernel_sample.shape)
        y_sample = tf.nn.conv2d(x_sample,kernel_sample,[1,1,1,1],"SAME")
        return y_sample
    x_values = np.random.randn(4,3, 5,5).astype('float32')
    kernel_values = np.random.randn(2,2,3,2).astype('float32')

    x = tf.constant(x_values)
    kernel = tf.constant(kernel_values)
    y = tf.nn.depthwise_conv2d(x,kernel,[1,1,1,1],"SAME", data_format='NCHW')

    kernel_scan = tf.transpose(kernel,[2,0,1,3])
    kernel_scan = tf.expand_dims(kernel_scan,axis=3) # [in_c, k, k, 1, out_c]

    x_scan = tf.transpose(x,[1, 0, 2, 3])
    x_scan = tf.expand_dims(x_scan,axis=1) # [in_c, batch, s, s, 1]
    print(x_scan.shape, kernel_scan.shape)
    c = tf.scan(f, (x_scan,kernel_scan) ,initializer = tf.zeros((4,5,5,2)))

    c = tf.transpose(c,[1, 0, 4, 2,3])
    c = tf.reshape(c,[4,6, 5,5])
    with tf.Session() as sess:
         y_values,kernel_scan_values,x_scan_values,c_values  = sess.run([y,kernel_scan,x_scan,c])

    #check input shapes for the scanned version
    print(kernel_scan_values.shape)
    print(x_scan_values.shape)
    #check output shapes
    print(c_values.shape)
    print(y_values.shape)
    #check values using depthwise_conv2d or the scanned version
    print(y_values[0,0,0,:])
    print(c_values[0,0,0,:])


tf_scan_depthwise_conv2d()
raise

def scan_depthwise_3d(x, filter):
    def f(input):
        x_sample = input[0]
        kernel_sample = input[1]

        y_sample = tf.nn.conv3d(x_sample,kernel_sample,[1,1,1,1,1],"SAME")
        return y_sample


    in_shape = x.get_shape().as_list()

    kernel_scan = tf.transpose(filter, [3, 0, 1, 2, 4])
    kernel_scan = tf.expand_dims(kernel_scan, axis=-1)
    x_scan = tf.transpose(x, [4, 0, 1, 2, 3])
    x_scan = tf.expand_dims(x_scan, axis=-1)
    c = tf.scan(f, (x_scan, kernel_scan), initializer=tf.zeros(shape=tf.shape(x_scan)))
    c = tf.transpose(c, [1, 2, 3, 4, 0, 5])
    c = tf.reshape(c,[4,5,5,5,6])

    with tf.Session() as sess:
         kernel_scan_values,x_scan_values,c_values  = sess.run([kernel_scan,x_scan,c])


    print(kernel_scan_values.shape)
    print(x_scan_values.shape)
    print(c_values.shape)
    print(c_values[0,0,0,0,:])


if __name__ == '__main__':
    # x = tf.random.normal(shape=(16, 256, 16, 64, 64))
    # y = conv3d(x, 256, 3, 'leaky_relu', param=0.2)
    # print(y.shape)

    # loss = tf.reduce_sum(y)
    # optim = tf.train.GradientDescentOptimizer(1e-4)
    # backward = optim.minimize(loss)

    n = 20
    # with tf.Session(config=config) as sess:

    #     sess.run(tf.global_variables_initializer())

    #     start = time.time()
    #     for i in tqdm.tqdm(range(n)):
    #         if i == n - 1:
    #             pid = os.getpid()
    #             py = psutil.Process(pid)
    #             print(f"CPU Percent: {py.cpu_percent()}")
    #             print(f"Memory info: {py.memory_info().rss / 1024 ** 2}")

    #         _, l = sess.run([backward, loss])
    #     end = time.time()
    #     print(f"Conv duration: {end - start}")

    # tf.reset_default_graph()
    x = tf.random.normal(shape=(16, 256, 16, 64, 64))
    with tf.variable_scope('gconv'):
        out = group_conv3d(x, 256, 3, groups=x.shape[1], activation='leaky_relu', param=0.2)

        print(out.shape)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        start = time.time()
        for i in tqdm.tqdm(range(n)):
            if i == n - 1:
                pid = os.getpid()
                py = psutil.Process(pid)
                print(f"CPU Percent: {py.cpu_percent()}")
                print(f"Memory info: {py.memory_info().rss / 1024 ** 2}")

            sess.run(out)
        end = time.time()
        print(f"Depthwise duration: {end - start}")



