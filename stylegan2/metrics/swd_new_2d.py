import numpy as np
import scipy.ndimage
from utils import uniform_box_sampler

#----------------------------------------------------------------------------

def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape # (minibatch, channel, height, width)
    assert len(S) == 4
    N = nhoods_per_image * S[0]
    H = nhood_size[0] // 2
    W = nhood_size[1] // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:S[1], -H:H+1, -W:W+1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(W, S[3] - W, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]

#----------------------------------------------------------------------------

def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4 # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc

#----------------------------------------------------------------------------

def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    assert A.ndim == 2 and A.shape == B.shape                           # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)             # (descriptor_component, direction)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True)) # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)                                      # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(projA, axis=0)                                  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)                                   # pointwise wasserstein distances
        results.append(np.mean(dists))                                  # average over neighborhoods and directions
    return np.mean(results)                                             # average over repeats

#----------------------------------------------------------------------------

def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (t[:, :, 0::2, 0::2] + t[:, :, 0::2, 1::2] + t[:, :, 1::2, 0::2] + t[:, :, 1::2, 1::2]) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)

#----------------------------------------------------------------------------

gaussian_filter = np.float32([
    [1, 4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4,  6,  4,  1]]) / 256.0

def pyr_down(minibatch): # matches cv2.pyrDown()
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode='mirror')[:, :, ::2, ::2]

def pyr_up(minibatch): # matches cv2.pyrUp()
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')

def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid

def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch


def get_swd_for_images(images1, images2, nhood_size=(7, 7), nhoods_per_image=128, dir_repeats=4, dirs_per_repeat=128):

    resolutions = []
    res = images1.shape[-1]

    while res >= 16:
        resolutions.append(res)
        res //= 2

    descriptors_real = [[] for res in resolutions]
    descriptors_fake = [[] for res in resolutions]

    for lod, level in enumerate(generate_laplacian_pyramid(images1, len(resolutions))):
        desc = get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image)
        descriptors_real[lod].append(desc)

    for lod, level in enumerate(generate_laplacian_pyramid(images2, len(resolutions))):
        desc = get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image)
        descriptors_fake[lod].append(desc)

    descriptors_real = [finalize_descriptors(d) for d in descriptors_real]
    descriptors_fake = [finalize_descriptors(d) for d in descriptors_fake]

    dist = [sliced_wasserstein(dreal, dfake, dir_repeats, dirs_per_repeat) for dreal, dfake in zip(descriptors_real, descriptors_fake)]

    dist = [d * 1e3 for d in dist] # multiply by 10^3

    dist = dist + [np.mean(dist)]

    return dist


if __name__ == '__main__':
#----------------------------------------------------------------------------

    shape = (128, 1, 128, 128)
    const_batch = np.full(shape=shape, fill_value=.05).astype(np.float32) + np.random.randn(*shape) * 1e-6
    rand_batch = np.random.rand(*shape)
    black_noise = const_batch + np.random.randn(*const_batch.shape) * .01

    noise_black_patches = rand_batch.copy()
    for _ in range(8):
        arr_slices = uniform_box_sampler(noise_black_patches, min_width=(32, 1, 6, 6,), max_width=(32, 1, 12, 12))[0]
        noise_black_patches[arr_slices] = 0

    print("black/black", get_swd_for_images(const_batch, const_batch, ))
    print("rand/rand", get_swd_for_images(rand_batch, rand_batch, ))
    print('black/rand', get_swd_for_images(const_batch, rand_batch, ))
    print('black/black+noise', get_swd_for_images(const_batch, black_noise, ))
    print('rand/black+noise', get_swd_for_images(rand_batch, black_noise, ))
    print('rand/rand+blackpatches', get_swd_for_images(rand_batch, noise_black_patches, ))
    print('black/rand+blackpatches', get_swd_for_images(const_batch, noise_black_patches, ))

