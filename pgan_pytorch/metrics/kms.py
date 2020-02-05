import numpy as np


def kolmogorov_smirnov_distance(real_images, fake_images, intercept, clip_range):
    # To Do: across the laplacian pyramid.
    
    real_images = ((real_images * intercept) + intercept).astype(int)
    fake_images = ((fake_images * intercept) + intercept).astype(int)
    real_images = real_images.clip(*clip_range)
    fake_images = fake_images.clip(*clip_range)
    
    fake_images = fake_images.mean(1)
    real_images = real_images.mean(1)

    real_images = real_images.reshape(real_images.shape[0], -1)
    fake_images = fake_images.reshape(real_images.shape[0], -1)

    real_hists = np.stack([np.histogram(real_images[i], bins=clip_range[1] - clip_range[0], density=True)[0] for i in range(real_images.shape[0])])
    fake_hists = np.stack([np.histogram(fake_images[i], bins=clip_range[1] - clip_range[0], density=True)[0] for i in range(fake_images.shape[0])])
    
    real_dist = real_hists.mean(0)
    fake_dist = fake_hists.mean(0)

    return abs(real_dist - fake_dist).max()
    