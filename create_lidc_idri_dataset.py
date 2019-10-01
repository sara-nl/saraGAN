import torch
import os
from functools import partial
import SimpleITK as sitk
import numpy as np
import argparse
from multiprocessing import Pool
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import h5py
import sys


def get_dcm_paths(root):
    for (directory, subdirectories, files) in os.walk(root):
        if any(path.endswith('.dcm') for path in os.listdir(directory)):
            yield directory


def read_dcm_series(path):
    if not os.path.isdir(path):
        raise ValueError(f'{path} is not a directory')

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    sitk_image = reader.Execute()

    return sitk_image


def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

# https://dcm_pathsthub.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def resample_sitk_image(sitk_image, spacing=None, interpolator=None,
                        fill_value=0):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """
    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()
    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1: #  8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing]*num_dim
    else:
        new_spacing = [float(s) if s else orig_spacing[idx] for idx, s in enumerate(spacing)]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(),\
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size*(orig_spacing/new_spacing)
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    # SimpleITK expects lists, not ndarrays
    new_size = [int(s) if spacing[idx] else int(orig_size[idx]) for idx, s in enumerate(new_size)]

    resample_filter = sitk.ResampleImageFilter()
    resampled_sitk_image = resample_filter.Execute(
        sitk_image,
        new_size,
        sitk.Transform(),
        sitk_interpolator,
        orig_origin,
        new_spacing,
        orig_direction,
        fill_value,
        orig_pixelid
    )

    return resampled_sitk_image, orig_spacing


def L(x, a):
    return np.sinc(x) * np.sinc(x / a)

def lanczos_3d(x, axis, a=4):
    
    assert x.shape[-3] == x.shape[-2] == x.shape[-1]
    # X-axis
    filter_size = x.shape[-1]
    d = filter_size / 2  # Location to interpolate on.
    ds = np.arange(-d + 0.5, d - 0.5 + 1)  # Distances to consider.
    l_in = ds * (a / d)  # Normalize to Lanczos a range.
    f = L(l_in, a)
    f = f[:, np.newaxis, np.newaxis] * f[np.newaxis, :, np.newaxis] * f[np.newaxis, np.newaxis, :]  # Make 3d
    
    f = f / f.sum()  # normalize
    for _ in range(len(f.shape), len(x.shape)):
        f = f[np.newaxis, ...]
        
    filtered = (f * x).sum(axis=axis)
    
    return filtered

def read_resample_resize_dcm(path, reduce_fn):

    image = read_dcm_series(path)

    metadata = {}

    metadata['path'] = path
    metadata['orig_depth'] = image.GetDepth()
    metadata['orig_spacing'] = tuple(image.GetSpacing())
    metadata['orig_origin'] = tuple(image.GetOrigin())
    metadata['orig_direction'] = tuple(image.GetDirection())
    metadata['orig_size'] = tuple(image.GetSize())

    if metadata['orig_size'][0] != 512:
        raise RuntimeError(f"Not a 512x512 image: {metadata['orig_size']}")
    
    if metadata['orig_spacing'][-1] > 3:
        raise RuntimeError(f"Spacing too large: {metadata['orig_spacing']}")

    elif metadata['orig_direction'] != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        raise RuntimeError("Not the proper direction.")

    array = sitk.GetArrayFromImage(image)
    metadata['orig_min'] = array.min()
    metadata['orig_max'] = array.max()

    # Resampled z-axis with linear interpolation
#     new_spacing = tuple(np.array(metadata['orig_spacing']) * np.array(metadata['orig_size']) / 512)
    new_spacing = (1, 1, 3)
    resampled_image, orig_spacing = resample_sitk_image(image, spacing=new_spacing, interpolator='linear')
    metadata['resampled_depth'] = resampled_image.GetDepth()
    metadata['resampled_spacing'] = tuple(resampled_image.GetSpacing())
    metadata['resampled_origin'] = tuple(resampled_image.GetOrigin())
    metadata['resampled_direction'] = tuple(resampled_image.GetDirection())
    
    resampled_array = sitk.GetArrayFromImage(resampled_image)
    metadata['resampled_size'] = tuple(resampled_array.shape)    
    if metadata['resampled_size'][0] > 160:
        raise RuntimeError(f"Z-dim too large: {metadata['resampled_size']}")
    
    pad_value = -1024
    clip_value = 2048
    resampled_array = np.clip(resampled_array, pad_value, clip_value)
    
    shape = resampled_array.shape
    x_pad = (512 - resampled_array.shape[2]) / 2 # 4 * 2 ** 7
    y_pad = (512 - resampled_array.shape[1]) / 2 # 4 * 2 ** 7
    z_pad = (128 - resampled_array.shape[0]) # 1.25 * 2 ** 7
        
    metadata['resampled_min'] = resampled_array.min()
    metadata['resampled_max'] = resampled_array.max()
    
    resampled_array = resampled_array - pad_value
    resampled_array = np.pad(resampled_array, [(0, 0), (int(np.floor(y_pad)), int(np.ceil(y_pad))), (int(np.floor(x_pad)), int(np.ceil(x_pad)))], constant_values=0, mode='constant')
        
    if resampled_array.shape[0] > 128:
        resampled_array = resampled_array[resampled_array.shape[0] - 128:, :, :]
    else:
        resampled_array = np.pad(resampled_array, [(z_pad, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
    assert resampled_array.shape == (128, 512, 512), resampled_array.shape
    
    resampled_arrays = [resampled_array.astype(np.uint16)]

    for i in range(1, 8):
        kernel = 2 ** i
        reduced = block_reduce(resampled_array, (kernel, kernel, kernel), func=reduce_fn, cval=0)
        reduced = np.clip(reduced, 0, clip_value - pad_value)
        reduced = reduced.astype(np.uint16)
        resampled_arrays.append(reduced)
    metadata['intercept'] = abs(pad_value)
    metadata['data_shape'] = 'DHW'

    return resampled_arrays, metadata


def get_dicom_iterator(root, reduce_fn):
    for path in get_dcm_paths(root):
        try:
            array, metadata = read_resample_resize_dcm(path, reduce_fn=reduce_fn)
        except RuntimeError as e:
            print(e)
            print("Continuing...")
            continue
        else:
            yield array, metadata

reduce_fn = lanczos_3d
dataset_dir = f'/project/davidr/lidc_idri/'
hdf5_dir = os.path.join(dataset_dir, 'hdf5', reduce_fn.__name__)

if not os.path.exists(hdf5_dir):
    os.makedirs(hdf5_dir)
    
hdf5_files = None

total = len(os.listdir(dataset_dir))
num_iters = total

for i, (arrays, metadata) in enumerate(get_dicom_iterator(dataset_dir, reduce_fn)):
    
    if hdf5_files is None:
        
        hdf5_files = {}
        intercept = np.array(metadata['intercept'])
        
        for array in arrays:
            size = array.shape[-1]
            dataset_shape = [num_iters] + list(array.shape)
            hdf5_file = h5py.File(os.path.join(hdf5_dir, f'{size}x{size}.h5'), mode='w')
            hdf5_file.create_dataset('data', dataset_shape, np.uint16)
            hdf5_file.create_dataset('intercept', [num_iters] + list(intercept.shape), np.int)
            hdf5_files[f'{size}x{size}'] = hdf5_file
    
    for array in arrays:
        size = array.shape[-1]
        hdf5_files[f'{size}x{size}']['data'][i, ...] = array[None]
        hdf5_files[f'{size}x{size}']['intercept'][i, ...] = intercept
        
        pt_dir = os.path.join(dataset_dir, 'pt', reduce_fn.__name__, f'{size}x{size}')
        npy_dir = os.path.join(dataset_dir, 'npy', reduce_fn.__name__, f'{size}x{size}')
        
        if not os.path.exists(pt_dir):
            os.makedirs(pt_dir)

        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)

        torch.save(torch.from_numpy((array - intercept).astype(np.int16)), os.path.join(pt_dir, f'{i:04}.pt'))
        np.save(os.path.join(npy_dir, f'{i:04}.npy'), array)
        
    print(i, num_iters)

for size in hdf5_files:
    hdf5_files[size].close()
