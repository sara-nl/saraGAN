import os
import matplotlib.pyplot as plt
import tensorflow as tf


def sample_images(generator, latent_dim, dataset_name, phase, step):
    r, c = 5, 5
    noise = tf.random.normal(shape=(r * c, 1, 1, latent_dim))
    gen_imgs = generator(noise).numpy()

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            img = gen_imgs[cnt, :, :, :]
            if img.shape[-1] == 1:
                img = img.squeeze()
                axs[i, j].imshow(img, cmap='gray')
            else:
                axs[i, j].imshow(img)
            axs[i, j].axis('off')
            cnt += 1

    save_dir = os.path.join('images', dataset_name, str(phase))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig.savefig(os.path.join(save_dir, f'{step:04}.png'))
    plt.close()

