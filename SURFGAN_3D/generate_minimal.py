import os
import argparse
import json
import importlib

import tensorflow as tf
import numpy as np

import dataset as data

from utils import get_num_phases, get_current_input_shape, get_base_shape

def main(args, config):

    phase = args.phase

    logdir = os.path.join(args.output_dir, 'generated_images')
    os.makedirs(logdir, exist_ok=True)

    print("Arguments passed:")
    print(args)
    print(f"Saving files to {logdir}")

    with tf.variable_scope('alpha'):
        alpha = tf.Variable(0, name='alpha', dtype=tf.float32)

    real_image_input = tf.placeholder(shape=get_current_input_shape(args.phase, args.batch_size, args.start_shape), dtype=tf.float32)
    z = tf.random.normal(shape=[tf.shape(real_image_input)[0], args.latent_dim])
    gen_sample = generator(z, alpha, args.phase, get_num_phases(args.start_shape, args.final_shape),
                           args.first_conv_nfilters, get_base_shape(args.start_shape), activation=args.activation, kernel_shape=args.kernel_shape, kernel_spec = args.kernel_spec, filter_spec = args.filter_spec, 
                           param=args.leakiness, size=args.network_size, is_reuse=False)

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        trainable_variable_names = [v.name for v in tf.trainable_variables()]
        var_names = [v.name for v in gen_vars]
        load_vars = [sess.graph.get_tensor_by_name(n) for n in var_names if n in trainable_variable_names]
        saver = tf.train.Saver(load_vars)
        print("Restoring variables...")
        saver.restore(sess, os.path.join(args.model_path))

        fake_batch = []
        # Fake images are always generated with the batch size used for training
        # Here, we loop often enough to make sure we have enough samples for the batch size that we want to use for metric computation
        fake_batch = sess.run(gen_sample).astype(np.float32)
        while fake_batch.shape[0] < args.num_samples:
            fake_batch = np.concatenate((fake_batch, sess.run(gen_sample).astype(np.float32)))

        fake_batch = data.invert_normalize_numpy(fake_batch, args.data_mean, args.data_stddev, True)

        print(f"fake_batch.shape = {fake_batch.shape}")

        np.save(os.path.join(logdir, 'fake_images.npy'), fake_batch)

def kernel_spec(value):
    with open(value) as json_file:
        data = json.load(json_file)
    return data['kernel_spec']
def filter_spec(value):
    with open(value) as json_file:
        data = json.load(json_file)
    return data['filter_spec']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str)
    parser.add_argument('--start_shape', type=str, default=None, required=True, help="Shape of the data at phase 0, '(c, z, y, x)', e.g. '(1, 5, 16, 16)'")
    parser.add_argument('--final_shape', type=str, default=None, required=True, help="'(c, z, y, x)', e.g. '(1, 64, 128, 128)'")
    parser.add_argument('--kernel_spec', type=kernel_spec, default = None, help = "A kernel specification file (in JSON) that lists the convolutional kernel shapes to be used in each layer. The JSON file should define this under the 'kernel_spec' keyword.")
    parser.add_argument('--filter_spec', type=filter_spec, default = None, help = "A specification file (in JSON) that lists the amount of filters to be used in each layer. The JSON file should define this under the 'filter_spec' keyword. Note that the kernel_spec and filter_spec may be stored in the same JSON, but you will have to supply both arguments.")
    parser.add_argument('--network_size', default=None, choices=['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl'], required=True)
    parser.add_argument('--latent_dim', type=int, default=None, required=True)
    parser.add_argument('--first_conv_nfilters', type=int, default=None, required=True, help='Number of filters in the first convolutional layer. Since it is densely connected to the latent space, the number of connections can increase rapidly, hence it can be set separately from the other filter counts deeper in the network')
    parser.add_argument('--kernel_shape', default=[3,3,3])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None, required=True, help='Path to the Tensorflow model checkpoint')
    parser.add_argument('--num_samples', default=None, type=int, required=True)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--phase', type=int, default=None, required=True)
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_mean', default=None, type=float, required=False, help="Mean of the input data. Used for input normalization. E.g. in the case of CT scans, this would be the mean CT value over all scans. Note: normalization is only performed if both data_mean and data_stddev are defined.")
    parser.add_argument('--data_stddev', default=None, type=float, required=False, help="Standard deviation of the input data. Used for input normalization. E.g. in the case of CT scans, this would be the standard deviation of CT values over all scans. Note: normalization is only performed if both data_mean and data_stddev are defined.")
    args = parser.parse_args()

    # TF Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    gopts = tf.GraphOptions(place_pruned_graph=True)
    config = tf.ConfigProto(graph_options=gopts,
                            intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']),
                            inter_op_parallelism_threads=2,
                            allow_soft_placement=True,
                            device_count={'CPU': int(os.environ['OMP_NUM_THREADS'])})

    # Get the right generator (right architecture)
    generator = importlib.import_module(f'networks.{args.architecture}.generator').generator

    main(args, config)