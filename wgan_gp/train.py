import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import horovod.tensorflow as hvd
import argparse


from loss import WassersteinLoss, GradientPenaltyLoss
from networks import build_generator, build_discriminator
from load_data import load_data
from utils import sample_images

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


@tf.function
def train_generator(generator, discriminator, generator_optim, batch_size, latent_dim, w_loss, is_first_batch):
    with tf.GradientTape() as tape:
        z = tf.random.normal(shape=(batch_size, 1, 1, latent_dim))
        x_fake = generator(z)
        d_fake = discriminator(x_fake)
        g_loss = -1 * w_loss(d_fake)
        # g_loss = g_loss_fn(d_fake)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(g_loss, generator.trainable_variables)
    generator_optim.apply_gradients(zip(grads, generator.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if is_first_batch:
        hvd.broadcast_variables(generator.variables, root_rank=0)
        hvd.broadcast_variables(generator_optim.variables(), root_rank=0)

    return g_loss


@tf.function
def train_discriminator(generator, discriminator, discriminator_optim, batch_size, latent_dim,
                        gp_loss, w_loss, x_real, is_first_batch):

    with tf.GradientTape() as tape:
        z = tf.random.normal(shape=(batch_size, 1, 1, latent_dim))
        x_fake = generator(z)
        d_fake = discriminator(x_fake)
        d_real = discriminator(x_real)

        # d_loss = d_loss_fn(d_real, d_fake)

        gradient_penalty_loss = gp_loss(discriminator, x_real, x_fake)
        # gradient_penalty_loss = gradient_penalty(discriminator, x_real, x_fake, mode='wgan-gp')

        # d_loss = sum(d_loss) + gradient_penalty_loss
        d_loss = w_loss(d_fake) - w_loss(d_real) + 10 * gradient_penalty_loss

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optim.apply_gradients(zip(grads, discriminator.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if is_first_batch:
        hvd.broadcast_variables(discriminator.variables, root_rank=0)
        hvd.broadcast_variables(discriminator_optim.variables(), root_rank=0)

    return d_loss


def train(train_data, dataset_name, latent_dim, n_critic, epoch_size, epochs, learning_rate, sample_interval):

    images, labels = next(iter(train_data.take(1)))
    batch_size, image_width, image_height, image_channels = images.shape

    generator = build_generator(latent_dim=latent_dim,
                                output_channels=image_channels)
    # generator.build_graph((batch_size, image_width, image_height, latent_dim))
    if hvd.rank() == 0:
        generator.summary()

    discriminator = build_discriminator(input_shape=(image_width, image_height, image_channels),
                                        latent_dim=latent_dim)
    # discriminator.build_graph((batch_size, image_width, image_height, image_channels))
    if hvd.rank() == 0:
        discriminator.summary()

    w_loss = WassersteinLoss()
    gp_loss = GradientPenaltyLoss()

    # Horovod: adjust learning rate based on number of GPUs.
    generator_optim = tf.optimizers.Adam(learning_rate * hvd.size(), beta_1=0, beta_2=0.99, epsilon=1e-8)
    discriminator_optim = tf.optimizers.Adam(learning_rate * hvd.size(), beta_1=0, beta_2=0.99, epsilon=1e-8)

    checkpoint_dir = './checkpoints'
    checkpoint_generator = tf.train.Checkpoint(model=generator, optimizer=generator_optim)
    checkpoint_discriminator = tf.train.Checkpoint(model=discriminator, optimizer=discriminator_optim)

    # Horovod: adjust number of steps based on number of GPUs.
    num_epoch_steps = epoch_size // batch_size // hvd.size()

    g_loss = None  # Placeholder

    for epoch in range(1, epochs + 1):

        if hvd.rank() == 0:
            print(f'\n Epoch {epoch} \n')

        sample_images(generator, latent_dim=latent_dim, dataset_name=dataset_name, phase=1, step=epoch)
        for batch, (images, labels) in enumerate(train_data.take(num_epoch_steps)):
            is_first_batch = batch == 0
            d_loss = train_discriminator(generator,
                                         discriminator,
                                         discriminator_optim,
                                         batch_size,
                                         latent_dim,
                                         gp_loss,
                                         w_loss,
                                         images,
                                         is_first_batch)

            if batch % n_critic == 0:
                g_loss = train_generator(generator,
                                         discriminator,
                                         generator_optim,
                                         batch_size,
                                         latent_dim,
                                         w_loss,
                                         is_first_batch)

            if batch % sample_interval == 0 and hvd.local_rank() == 0:
                print(f'Batch {batch:03} \t Discriminator Loss: {d_loss:.4f} \t Generator Loss: {g_loss:.4f}')

        # Horovod: save checkpoints only on worker 0 to prevent other workers from
        # corrupting it.
        if hvd.rank() == 0:
            checkpoint_discriminator.save(checkpoint_dir)
            checkpoint_generator.save(checkpoint_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--n_critic', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--epoch-size', type=int, default=50000)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--epochs-per-phase', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sample_interval', type=int, default=25, help='How often to sample and save images.')

    args = parser.parse_args()

    dataset = load_data(args.dataset)

    train(dataset, args.dataset, args.latent_dim, args.n_critic, args.epoch_size,
          args.epochs, args.learning_rate, args.sample_interval)
