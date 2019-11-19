import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import random_ops

from utils import uniform_box_sampler

filter_1d = [1, 4, 6, 4, 1]
f = np.array(filter_1d, dtype=np.float32)
f = f[:, np.newaxis, np.newaxis] * f[np.newaxis, np.newaxis, :] * f[np.newaxis, :, np.newaxis]
gaussian_filter = f / f.sum()
_GAUSSIAN_FILTER = gaussian_filter.reshape(5, 5, 5, 1, 1)


def _normalize_patches(patches):
    """Normalize patches by their mean and standard deviation.
  Args:
    patches: (tensor) The batch of patches (batch, size, size, channels).
  Returns:
    Tensor (batch, size, size, channels) of the normalized patches.
  """
    patches = array_ops.concat(patches, 0)
    mean, variance = nn.moments(patches, [1, 2, 3, 4], keepdims=True)
    patches = (patches - mean) / math_ops.sqrt(variance)
    return array_ops.reshape(patches, [array_ops.shape(patches)[0], -1])


def _laplacian_pyramid(batch, num_levels):
    """Compute a Laplacian pyramid.
  Args:
    batch: (tensor) The batch of images (batch, height, width, channels).
    num_levels: (int) Desired number of hierarchical levels.
  Returns:
    List of tensors from the highest to lowest resolution.
  """
    gaussian_filter = constant_op.constant(_GAUSSIAN_FILTER)

    def spatial_conv(batch, gain):
        s = array_ops.shape(batch)
        padded = array_ops.pad(batch, [[0, 0], [2, 2], [2, 2], [2, 2], [0, 0]], 'REFLECT')
        xt = array_ops.transpose(padded, [0, 4, 1, 2, 3])
        xt = array_ops.reshape(xt, [s[0] * s[4], s[1] + 4, s[2] + 4, s[3] + 4, 1])
        conv_out = nn_ops.conv3d(xt, gaussian_filter * gain, [1] * 5, 'VALID')
        conv_xt = array_ops.reshape(conv_out, [s[0], s[4], s[1], s[2], s[3]])
        conv_xt = array_ops.transpose(conv_xt, [0, 2, 3, 4, 1])
        return conv_xt

    def pyr_down(batch):  # matches cv2.pyrDown()
        print(batch.shape)
        return spatial_conv(batch, 1)[:, ::2, ::2, ::2]

    def pyr_up(batch):  # matches cv2.pyrUp()
        s = array_ops.shape(batch)
        zeros = array_ops.zeros([7 * s[0], s[1], s[2], s[3], s[4]])
        res = array_ops.concat([batch, zeros], 0)
        res = array_ops.batch_to_space_nd(res, block_shape=[2, 2, 2],
                                          crops=[[0, 0], [0, 0], [0, 0]])
        res = spatial_conv(res, 4)
        return res

    pyramid = [math_ops.cast(batch, dtypes.float32)]
    for _ in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid


def _batch_to_patches(batch, patches_per_image, patch_size: tuple):
    """Extract patches from a batch.
  Args:
          batch: (tensor) The batch of images (batch, height, width, channels).
          patches_per_image: (int) Number of patches to extract per image.
          patch_size: (int) Size of the patches (size, size, channels) to extract.
  Returns:
          Tensor (batch*patches_per_image, patch_size, patch_size, channels) of
          patches.
  """

    def py_func_random_patches(batch):
        """Numpy wrapper."""
        batch_size, depth, height, width, channels = batch.shape
        patch_count = patches_per_image * batch_size
        ds = patch_size[0] // 2
        hs = patch_size[1] // 2
        ws = patch_size[2] // 2

        print(batch.shape)
        print(ds, depth, depth - ds)

        # Randomly pick patches.
        patch_id, z, y, x, chan = np.ogrid[0:patch_count, -ds:ds + 1, -hs:hs + 1, -ws: ws + 1,
                                  0:channels]
        img_id = patch_id // patches_per_image
        # pylint: disable=g-no-augmented-assignment
        # Need explicit addition for broadcast to work properly.
        z = z + np.random.randint(ds, depth - ds, size=(patch_count, 1, 1, 1, 1))
        y = y + np.random.randint(hs, height - hs, size=(patch_count, 1, 1, 1, 1))
        x = x + np.random.randint(ws, width - ws, size=(patch_count, 1, 1, 1, 1))
        # pylint: enable=g-no-augmented-assignment
        idx = (((img_id * height + y) * width + x) * depth + z) * channels + chan
        patches = batch.flat[idx]
        return patches

    patches = script_ops.py_func(
        py_func_random_patches, [batch], batch.dtype)
    return patches


def _sort_rows(matrix, num_rows):
    """Sort matrix rows by the last column.
  Args:
          matrix: a matrix of values (row,col).
          num_rows: (int) number of sorted rows to return from the matrix.
  Returns:
          Tensor (num_rows, col) of the sorted matrix top K rows.
  """
    tmatrix = array_ops.transpose(matrix, [1, 0])
    sorted_tmatrix = nn_ops.top_k(tmatrix, num_rows)[0]
    return array_ops.transpose(sorted_tmatrix, [1, 0])


def _sliced_wasserstein(a, b, random_sampling_count, random_projection_dim):
    """Compute the approximate sliced Wasserstein distance.
  Args:
          a: (matrix) Distribution "a" of samples (row, col).
          b: (matrix) Distribution "b" of samples (row, col).
          random_sampling_count: (int) Number of random projections to average.
          random_projection_dim: (int) Dimension of the random projection space.
  Returns:
          Float containing the approximate distance between "a" and "b".
  """
    s = array_ops.shape(a)
    means = []
    for _ in range(random_sampling_count):
        # Random projection matrix.
        proj = random_ops.random_normal(
            [array_ops.shape(a)[1], random_projection_dim])
        proj *= math_ops.rsqrt(
            math_ops.reduce_sum(math_ops.square(proj), 0, keepdims=True))
        # Project both distributions and sort them.
        proj_a = math_ops.matmul(a, proj)
        proj_b = math_ops.matmul(b, proj)
        proj_a = _sort_rows(proj_a, s[0])
        proj_b = _sort_rows(proj_b, s[0])
        # Pairwise Wasserstein distance.
        wdist = math_ops.reduce_mean(math_ops.abs(proj_a - proj_b))
        means.append(wdist)
    return math_ops.reduce_mean(means)


def sliced_wasserstein_distance_3d(real_images,
                                   fake_images,
                                   resolution_min=16,
                                   patches_per_image=256,
                                   patch_size=(2, 8, 8),
                                   random_sampling_count=1,
                                   random_projection_dim=2 * 8 * 8 * 1,
                                   use_svd=False):
    """Compute the Wasserstein distance between two distributions of images.
  Note that measure vary with the number of images. Use 8192 images to get
  numbers comparable to the ones in the original paper.
  Args:
                  real_images: (tensor) Real images (batch, height, width, channels).
                  fake_images: (tensor) Fake images (batch, height, width, channels).
                  resolution_min: (int) Minimum resolution for the Laplacian pyramid.
                  patches_per_image: (int) Number of patches to extract per image per
                          Laplacian level.
                  patch_size: (int) Width of a square patch.
                  random_sampling_count: (int) Number of random projections to average.
                  random_projection_dim: (int) Dimension of the random projection space.
                  use_svd: experimental method to compute a more accurate distance.
  Returns:
                  List of tuples (distance_real, distance_fake) for each level of the
                  Laplacian pyramid from the highest resolution to the lowest.
                          distance_real is the Wasserstein distance between real images
                          distance_fake is the Wasserstein distance between real and fake images.
  Raises:
                  ValueError: If the inputs shapes are incorrect. Input tensor dimensions
                  (batch, height, width, channels) are expected to be known at graph
                  construction time. In addition height and width must be the same and the
                  number of colors should be exactly 3. Real and fake images must have the
                  same size.
  """
    real_images = tf.transpose(real_images, perm=(0, 2, 3, 4, 1))
    fake_images = tf.transpose(fake_images, perm=(0, 2, 3, 4, 1))
    res = real_images.shape[-2]
    # Select resolutions.
    resolution_full = int(res)
    resolution_min = min(resolution_min, resolution_full)
    resolution_max = resolution_full
    # Base loss of detail.
    resolutions = [
        2 ** i
        for i in range(
            int(np.log2(resolution_max)),
            int(np.log2(resolution_min)) - 1, -1)
    ]

    # Gather patches for each level of the Laplacian pyramids.
    patches_real, patches_fake, patches_test = (
        [[] for _ in resolutions] for _ in range(3))
    for lod, level in enumerate(
            _laplacian_pyramid(real_images, len(resolutions))):
        patches_real[lod].append(
            _batch_to_patches(level, patches_per_image, patch_size))
        patches_test[lod].append(
            _batch_to_patches(level, patches_per_image, patch_size))

    for lod, level in enumerate(
            _laplacian_pyramid(fake_images, len(resolutions))):
        patches_fake[lod].append(
            _batch_to_patches(level, patches_per_image, patch_size))

    for lod in range(len(resolutions)):
        for patches in [patches_real, patches_test, patches_fake]:
            patches[lod] = _normalize_patches(patches[lod])

    # Evaluate scores.
    scores = []
    for lod in range(len(resolutions)):
        print('255\t\t\t', lod)
#  _sliced_wasserstein(patches_real[lod], patches_test[lod],
#                                                random_sampling_count, random_projection_dim),
        if not use_svd:
            scores.append((
                           _sliced_wasserstein(patches_real[lod], patches_fake[lod],
                                               random_sampling_count, random_projection_dim)))
        else:
            raise NotImplementedError()
    return scores


def test():

    shape = (128, 1, 16, 64, 64)

    black_batch = np.zeros(shape=shape) + 1e-9 * np.random.randn(*shape)
    rand_batch = np.random.rand(*shape)
    black_noise = black_batch + np.random.randn(*black_batch.shape)


    noise_black_patches = rand_batch.copy()
    for _ in range(8):
        arr_slices = uniform_box_sampler(noise_black_patches, min_width=(32, 1, 2, 6, 6,), max_width=(32, 1, 4, 12, 12))[0]
        noise_black_patches[arr_slices] = 1e-9

    with tf.session() as sess:
        print("black/black", sliced_wasserstein_distance_3d(black_batch, black_batch)[0].eval())
        print("rand/rand", sliced_wasserstein_distance_3d(rand_batch, rand_batch)[0].eval())
        print('black/rand', sliced_wasserstein_distance_3d(black_batch, rand_batch)[0].eval())
        print('black/black+noise', sliced_wasserstein_distance_3d(black_batch, black_noise)[0].eval())
        print('rand/black+noise', sliced_wasserstein_distance_3d(rand_batch, black_noise)[0].eval())
        print('rand/rand+blackpatches', sliced_wasserstein_distance_3d(rand_batch, noise_black_patches)[0].eval())
        print('black/rand+blackpatches', sliced_wasserstein_distance_3d(black_batch, noise_black_patches)[0].eval())


if __name__ == '__main__':
    test()
