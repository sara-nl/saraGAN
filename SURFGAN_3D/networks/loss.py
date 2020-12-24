import tensorflow as tf


def forward_generator(generator,
                      discriminator,
                      real_image_input,
                      latent_dim,
                      alpha,
                      phase,
                      num_phases,
                      base_dim,
                      base_shape,
                      activation,
                      leakiness,
                      network_size,
                      loss_fn,
                      noise_stddev,
                      is_reuse=False
                      ):
    z = tf.random.normal(shape=[tf.shape(real_image_input)[0], latent_dim])
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation=activation,
                           param=leakiness, size=network_size, is_reuse=is_reuse)

    # Add instance noise to make training more stable. See e.g. https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * noise_stddev
    gen_sample = gen_sample + tf.random.normal(shape=tf.shape(gen_sample)) * noise_stddev

    # Generator training.
    disc_fake_g = discriminator(gen_sample, alpha, phase, num_phases, base_shape, base_dim, latent_dim,
                                activation=activation, param=leakiness, size=network_size, is_reuse=is_reuse)
    if loss_fn == 'wgan':
        gen_loss = -tf.reduce_mean(disc_fake_g)

    elif loss_fn == 'logistic':
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return gen_sample, gen_loss


def forward_discriminator(generator,
                          discriminator,
                          real_image_input,
                          latent_dim,
                          alpha,
                          phase,
                          num_phases,
                          base_dim,
                          base_shape,
                          activation,
                          leakiness,
                          network_size,
                          loss_fn,
                          gp_weight,
                          noise_stddev,
                          is_reuse=False,
                          ):
    z = tf.random.normal(shape=[tf.shape(real_image_input)[0], latent_dim])
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation=activation,
                           param=leakiness, size=network_size, is_reuse=is_reuse)

    # Add instance noise to make training more stable. See e.g. https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * noise_stddev
    gen_sample = gen_sample + tf.random.normal(shape=tf.shape(gen_sample)) * noise_stddev

    # Discriminator Training
    disc_fake_d = discriminator(tf.stop_gradient(gen_sample), alpha, phase, num_phases,
                                base_shape, base_dim, latent_dim, activation=activation, param=leakiness,
                                size=network_size, )
    disc_real = discriminator(real_image_input, alpha, phase, num_phases,
                              base_shape, base_dim, latent_dim, activation=activation, param=leakiness,
                              is_reuse=True, size=network_size, )

    gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = gamma * real_image_input + (1 - gamma) * tf.stop_gradient(gen_sample)
    gradients = tf.gradients(discriminator(interpolates, alpha, phase,
                                           num_phases, base_shape, base_dim, latent_dim,
                                           is_reuse=True, activation=activation,
                                           param=leakiness, size=network_size, ), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3, 4)))

    if loss_fn == 'wgan':
        gradient_penalty = (slopes - 1) ** 2
        gp_loss = gp_weight * gradient_penalty
        disc_loss = disc_fake_d - disc_real
        drift_loss = 1e-3 * disc_real ** 2
        disc_loss = tf.reduce_mean(disc_loss + gp_loss + drift_loss)

    elif loss_fn == 'logistic':
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        gp_loss = gp_weight * gradient_penalty
        disc_loss = tf.reduce_mean(tf.nn.softplus(disc_fake_d)) + tf.reduce_mean(
            tf.nn.softplus(-disc_real))
        disc_loss += gp_loss

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return disc_loss, gp_loss


def forward_simultaneous(generator,
                         discriminator,
                         real_image_input,
                         latent_dim,
                         alpha,
                         phase,
                         num_phases,
                         base_dim,
                         base_shape,
                         activation,
                         leakiness,
                         network_size,
                         loss_fn,
                         gp_weight,
                         noise_stddev,
                         conditioning=None
                         ):
    z = tf.random.normal(shape=[tf.shape(real_image_input)[0], latent_dim])
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation=activation,
                           param=leakiness, size=network_size, conditioning=conditioning)

    # Add instance noise to make training more stable. See e.g. https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * noise_stddev
    gen_sample = gen_sample + tf.random.normal(shape=tf.shape(gen_sample)) * noise_stddev
    
    # Discriminator Training
    disc_fake_d = discriminator(tf.stop_gradient(gen_sample), alpha, phase, num_phases,
                                base_shape, base_dim, latent_dim, activation=activation, param=leakiness,
                                size=network_size, conditioning=conditioning)
    disc_real = discriminator(real_image_input, alpha, phase, num_phases,
                              base_shape, base_dim, latent_dim, activation=activation, param=leakiness,
                              is_reuse=True, size=network_size, conditioning=conditioning)

    gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = gamma * real_image_input + (1 - gamma) * tf.stop_gradient(gen_sample)

    gradients = tf.gradients(discriminator(interpolates, alpha, phase,
                                           num_phases, base_shape, base_dim, latent_dim,
                                           is_reuse=True, activation=activation,
                                           param=leakiness, size=network_size, conditioning=conditioning), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3)))

    # Generator training.
    disc_fake_g = discriminator(gen_sample, alpha, phase, num_phases, base_shape, base_dim, latent_dim,
                                activation=activation, param=leakiness, size=network_size, is_reuse=True, conditioning=conditioning)

    if loss_fn == 'wgan':
        gradient_penalty = (slopes - 1) ** 2
        gp_loss = gp_weight * gradient_penalty
        disc_loss = disc_fake_d - disc_real
        drift_loss = 1e-3 * disc_real ** 2
        disc_loss = tf.reduce_mean(disc_loss + gp_loss + drift_loss)
        gen_loss = -tf.reduce_mean(disc_fake_g)

    elif loss_fn == 'logistic':
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        gp_loss = gp_weight * gradient_penalty
        disc_loss = tf.reduce_mean(tf.nn.softplus(disc_fake_d)) + tf.reduce_mean(
            tf.nn.softplus(-disc_real))
        disc_loss += gp_loss
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return gen_loss, disc_loss, gp_loss, gen_sample
