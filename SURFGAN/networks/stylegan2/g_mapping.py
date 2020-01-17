from networks.ops import *


def g_mapping(
        z,
        phase,
        mapping_layers=8,
        mapping_fmaps=512,
        mapping_lrmul=.01,
        activation='leaky_relu',
        act_param=0.2,
        is_reuse=False):

    with tf.variable_scope('g_mapping') as scope:
        if is_reuse:
            scope.reuse_variables()
        x = z * tf.rsqrt(tf.reduce_mean(tf.square(z), axis=1, keepdims=True) + 1e-8)

        # Mapping layers.
        latent_fmaps = z.get_shape().as_list()[1]
        for layer_idx in range(mapping_layers):
            fmaps = latent_fmaps if layer_idx == mapping_layers - 1 else mapping_fmaps
            with tf.variable_scope(f'dense_{layer_idx}'):
                x = dense(x, fmaps=fmaps, activation=activation, lrmul=mapping_lrmul, param=act_param)
                x = apply_bias(x)
                x = act(x, activation, param=act_param)

        with tf.variable_scope('broadcast_latents'):
            x = tf.tile(x[:, tf.newaxis], [1, phase * 3 - 2, 1])

        return x


if __name__ == '__main__':

    phase = 8
    latents_in = tf.random.normal(shape=[1, 1024])
    dlatents = g_mapping(latents_in, phase)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        disentangled_latents = sess.run(dlatents)
        print(disentangled_latents.shape, disentangled_latents.min(), disentangled_latents.max())

        print('Total synthesis generator variables:',
              sum(np.product(p.shape) for p in tf.trainable_variables('g_mapping')))
