from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity
import numpy as np
import itertools
from multiprocessing import Pool
import os
from functools import partial

def get_mean_squared_error(real, fake):
    return mean_squared_error(real, fake)


def get_normalized_root_mse(real, fake):
    return normalized_root_mse(real, fake, normalization='min-max')


def get_psnr(real, fake, data_range=3072):
    return peak_signal_noise_ratio(real, fake, data_range=data_range)


def get_ssim(real, fake, data_range=3):
    real = np.transpose(real, [0, 2, 3, 4, 1])
    fake = np.transpose(fake, [0, 2, 3, 4, 1])
    if real.shape[0] == 1:
        real = real[0, ...]
    if fake.shape[0] == 1:
        fake = fake[0, ...]
    ssims = []

    # This parallel section is a nice idea, but for some reason does not work. All workers in the pool get scheduled to one core for some reason...
    # Haven't been able to solve it, but can't spend more time on it now
    # num_procs = int(os.getenv('OMP_NUM_THREADS', 1))
    # if num_procs > 1:
    #     print(f"Launching {num_procs}")
    #     ssim = partial(structural_similarity, data_range=data_range, multichannel=True, gaussian_weights=True)
    #         #return structural_similarity(im1, im2, data_range=data_range, multichannel=True, gaussian_weights=True)
    #     p = Pool(num_procs)
    #     arglist = [(im1, im2) for (im1, im2) in zip(real,fake)]
    #     ssims = p.starmap(ssim, arglist)
    #     print(f"Parallel result: {np.mean(ssims)}")

    for (im1, im2) in zip(real,fake):
        ssims.append(structural_similarity(im1, im2, data_range=data_range, multichannel=True, gaussian_weights=True))
    np.mean(ssims)
    return ssims


if __name__ == '__main__':

    volume1 = (np.clip(np.random.normal(size=(1, 1, 16, 64, 64)), -1, 2) * 1024).astype(np.int16)
    volume2 = (np.clip(np.random.normal(size=(1, 1, 16, 64, 64)), -1, 2) * 1024).astype(np.int16)

    print(volume1.min(), volume2.max(), volume1.min(), volume2.max())

    print(get_mean_squared_error(volume1, volume2))
    print(get_normalized_root_mse(volume1, volume2))
    print(get_psnr(volume1, volume2))
    print(get_ssim(volume1, volume2))
