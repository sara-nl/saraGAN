import tensorflow as tf
import numpy as np

from networks.g_mapping import g_mapping
from networks.g_synthesis import g_synthesis


def generator(z, alpha, phase, num_phases, base_dim, base_shape, activation, is_training,
              param=None, psi=0.7, truncation_layers=8, beta=0.995, style_mixing_prob=0.9,
              is_reuse=False):

    with tf.variable_scope('generator') as scope:
        if is_reuse:
            scope.reuse_variables()

        d_z_avg = tf.get_variable('d_z_ag', shape=z.get_shape().as_list()[1], initializer=tf.initializers.zeros(),
                                  trainable=False)
        d_z = g_mapping(z, phase)

        if is_training:
            with tf.variable_scope('d_z_avg'):
                batch_avg = tf.reduce_mean(d_z[:, 0], axis=0)
                update_op = tf.assign(d_z_avg, beta * d_z_avg + (1 - beta) * batch_avg)
                with tf.control_dependencies([update_op]):
                    d_z = tf.identity(d_z)

        # Style regularization. Requires more compute as we need two mapping passes.
        if is_training and phase > 1:
            z_reg = tf.random_normal(tf.shape(z))
            d_z_reg = g_mapping(z_reg, phase, is_reuse=True)

            layer_idx = np.arange(phase)[np.newaxis, :, np.newaxis]

            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, phase, dtype=tf.int32),
                lambda: phase)

            d_z = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(d_z)), d_z, d_z_reg)

        # Apply truncation trick.
        with tf.variable_scope('truncation'):
            layer_idx = np.arange(phase)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float32)
            coefs = tf.where(layer_idx < truncation_layers, psi * ones, ones)
            d_z = coefs * d_z + (1 - coefs) * d_z_avg

        img_out = g_synthesis(d_z, alpha, phase, num_phases, base_dim, base_shape, activation, param)

        return img_out


if __name__ == '__main__':

    z_dim = 256
    num_phases = 8
    base_dim = 256
    base_shape = (1, 1, 4, 4)
    activation = 'leaky_relu'
    param = 0.2

    for phase in range(1, 4):
        tf.reset_default_graph()

        z = tf.random.normal(shape=(1, z_dim))
        alpha = tf.Variable(0.5, name='alpha', trainable=False)
        img_out = generator(z, alpha, phase, num_phases, base_dim, base_shape, activation,
                            is_training=True, param=param)

        print('Total generator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('generator')))

        print('Output shape:', img_out.shape)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            img_out = sess.run(img_out)
            print(img_out.min(), img_out.max(), img_out.shape)





