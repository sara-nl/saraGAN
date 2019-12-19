import tensorflow as tf
from networks.ops import get_weight, conv3d
import time
import tqdm
import psutil
import os
gopts = tf.GraphOptions(place_pruned_graph=True)
config = tf.ConfigProto(graph_options=gopts)
config.gpu_options.allow_growth = True


def group_conv3d(x, fmaps, kernel, groups, activation, param=None):

    assert kernel >= 1 and kernel % 2 == 1
    assert groups <= x.shape[1]
    assert fmaps >= groups

    fmaps = fmaps // groups

    def _group_conv3d(x):
        w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], activation, param=param)

        inputs = tf.split(x, groups, axis=1)
        filters = tf.split(w, groups, axis=-2)
        x = tf.concat(
            [tf.nn.conv3d(i, f,
                          strides=[1, 1, 1, 1, 1],
                          padding='SAME',
                          data_format='NCDHW')
             for i, f in zip(inputs, filters)], axis=1)

        # x = conv3d(x, fmaps, kernel, activation=activation, param=param)

        return x

    _group_conv3d = tf.contrib.layers.recompute_grad(_group_conv3d)

    with tf.variable_scope('recompute', use_resource=True):
        return _group_conv3d(x)


if __name__ == '__main__':

    x = tf.random.normal(shape=(16, 256, 16, 64, 64))
    y = group_conv3d(x, 256, 3, x.shape[1], 'leaky_relu', param=0.2)
    print(y.shape)

    loss = tf.reduce_sum(y)

    optim = tf.train.GradientDescentOptimizer(1e-4)
    backward = optim.minimize(loss)

    n = 20
    with tf.Session(config=config) as sess:

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())

        start = time.time()
        for i in tqdm.tqdm(range(n)):
            if i == n - 1:
                print(psutil.cpu_percent())
                print(dict(psutil.virtual_memory()._asdict()))
            _, l = sess.run([backward, loss], options=run_options)
            print(l)
        end = time.time()
        print(f"Depthwise Conv duration: {end - start}")

    opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
    # opts["min_bytes"] = 0
    # opts["min_micros"] = 0
    opts["select"] = ("bytes", "peak_bytes", "output_bytes",
                         "residual_bytes")
    g = tf.get_default_graph()
    tf.profiler.profile(g, run_meta=run_metadata, options=opts, cmd='scope')

    # Print memory info.
    pid = os.getpid()
    py = psutil.Process(pid)
    print(f"CPU Percent: {py.cpu_percent()}")
    print(f"Memory info: {py.memory_info().rss / 1024**2}")
