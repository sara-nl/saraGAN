import numpy as np
import scipy.ndimage
np.seterr(divide='ignore')

#----------------------------------------------------------------------------

def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape # (minibatch, channel, height, width)
    N = nhoods_per_image * S[0]
    D = nhood_size[1] // 2
    H = nhood_size[2] // 2
    W = nhood_size[3] // 2
    nhood, chan, z, x, y = np.ogrid[0:N, 0:nhood_size[0], -D:D+1, -H:H+1, -W:W+1]
    img = nhood // nhoods_per_image
    z = z + np.random.randint(D, S[2] - D, size=(N, 1, 1, 1))
    x = x + np.random.randint(W, S[4] - W, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    idx = (((img * S[1] + chan) * S[2] + z) * S[3] + y) * S[4] + x
    return minibatch.flat[idx]


#----------------------------------------------------------------------------

def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 5 # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3, 4), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3, 4), keepdims=True)
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

filter_1d = [1, 4, 6, 4, 1]
f = np.array(filter_1d, dtype=np.float32)
f = f[:, np.newaxis, np.newaxis] * f[np.newaxis, np.newaxis, :] * f[np.newaxis, :, np.newaxis]
gaussian_filter = f / f.sum()


def pyr_down(minibatch): # matches cv2.pyrDown()
    assert minibatch.ndim == 5
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :, :], mode='mirror')[:, :, ::2, ::2, ::2]

def pyr_up(minibatch): # matches cv2.pyrUp()
    assert minibatch.ndim == 5
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2, S[4] * 2), minibatch.dtype)
    res[:, :, ::2, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :, :] * 4.0, mode='mirror')


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


def sliced_wasserstein_distance(minibatch_real, minibatch_fake):

    resolutions = []
    res = minibatch_real.shape[-1]
    while res >= 32:
        resolutions.append(res)
        res //= 2

    resolutions

    descriptors_real = [[] for res in resolutions]
    descriptors_fake = [[] for res in resolutions]

    for lod, level in enumerate(generate_laplacian_pyramid(minibatch_real, len(resolutions))):
        desc = get_descriptors_for_minibatch(level, (1, 2, 8, 8), 128)
        descriptors_real[lod].append(desc)

    for lod, level in enumerate(generate_laplacian_pyramid(minibatch_fake, len(resolutions))):
        desc = get_descriptors_for_minibatch(level, (1, 2, 8, 8), 128)
        descriptors_fake[lod].append(desc)

    desc_real = [finalize_descriptors(d) for d in descriptors_real]
    desc_fake = [finalize_descriptors(d) for d in descriptors_fake]

    dist = [sliced_wasserstein(dreal, dfake, 4, 128) for dreal, dfake in zip(desc_real, desc_fake)]
    dist = [d * 1e3 for d in dist] # multiply by 10^3
    dist + [np.mean(dist)]
    return dist
