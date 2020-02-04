import tensorflow as tf
import os
from functools import partial
import SimpleITK as sitk
import numpy as np
import skimage
import argparse
from skimage.measure import block_reduce
from multiprocessing import Pool

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
    assert x.shape[3] == x.shape[-2] == x.shape[-1], "Cuboid filter not implemented yet."
    filter_size = x.shape[-1]
    d = filter_size / 2  # Location to interpolate on.
    ds = np.arange(-d + 0.5, d - 0.5 + 1)  # Distances to consider.
    l_in = ds * (a / d)  # Normalize to Lanczos a range.
    f = L(l_in, a)  # 1D filter
    f = f[:, np.newaxis, np.newaxis] * f[np.newaxis, :, np.newaxis] * f[np.newaxis, np.newaxis, :]  # Make 3d
    f = f / f.sum()  # normalize
    for _ in range(len(f.shape), len(x.shape)):
        f = f[np.newaxis, ...]
    
    return (f * x).sum(axis=axis)


def read_resample_resize_dcm(path):

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
    resampled_array = np.clip(resampled_array, pad_value, 2048)
    
    shape = resampled_array.shape
    x_pad = (512 - resampled_array.shape[2]) / 2 # 4 * 2 ** 7
    y_pad = (512 - resampled_array.shape[1]) / 2 # 4 * 2 ** 7
    z_pad = (128 - resampled_array.shape[0]) # 1.25 * 2 ** 7
        
    resampled_array = np.pad(resampled_array, [(0, 0), (int(np.floor(y_pad)), int(np.ceil(y_pad))), (int(np.floor(x_pad)), int(np.ceil(x_pad)))], constant_values=pad_value)
    
    if resampled_array.shape[0] > 128:
        resampled_array = resampled_array[resampled_array.shape[0] - 128:, :, :]
    else:
        resampled_array = np.pad(resampled_array, [(z_pad, 0), (0, 0), (0, 0)], constant_values=pad_value)
        
    assert resampled_array.shape == (128, 512, 512), resampled_array.shape
    resampled_array = resampled_array / abs(pad_value)

    metadata['resampled_min'] = resampled_array.min()
    metadata['resampled_max'] = resampled_array.max()

    resampled_arrays = [resampled_array]
    print(resampled_array.shape, resampled_array.min(), resampled_array.max())
    
    for i in range(1, 8):
        kernel = 2 ** i
        resampled_arrays.append(block_reduce(resampled_array, (kernel, kernel, kernel), func=lanczos_3d, cval=pad_value / abs(pad_value)))
    metadata['normalization_constant'] = abs(pad_value)
    metadata['data_shape'] = 'DHW'

    return resampled_arrays, metadata


def get_dicom_iterator(root):
    for path in get_dcm_paths(root):
        try:
            array, metadata = read_resample_resize_dcm(path)
        except RuntimeError as e:
            print(e)
            print("Continuing...")
            continue
        else:
            yield array, metadata

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def to_tfrecords(tensor_metadata, counter, output_dir):
    # write records to a tfrecords file
    writer = tf.io.TFRecordWriter(os.path.join(output_dir, f'{counter:04}.tfrecord'))

    tensor = tensor_metadata[0]
    metadata = tensor_metadata[1]

    # Loop through all the features you want to write
    features = {}
    features['image'] = tf.train.Feature(float_list=tf.train.FloatList(value=tensor.numpy().flatten()))
    for k in metadata:
        value = metadata[k].numpy()

        if metadata[k].dtype == tf.string:
            feature = _bytes_feature(value)
        elif metadata[k].dtype == tf.float32:
            feature = _float_feature(value.flatten())

        elif metadata[k].dtype == tf.int16:
            feature = _int64_feature(value.flatten())
        else:
            raise NotImplementedError(f"Unsupported dtype {metadata[k].dtype}")

        features[k] = feature

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=features))

    # Serialize the example to a string
    serialized = example.SerializeToString()

    # write the serialized objec to the disk
    writer.write(serialized)
    writer.close()


dataset_dir = '/project/davidr/lidc_idri/'
size = 64
# size = args.image_size

array_output_type = tf.float32
array_output_shape = (size, size, size)

metadata_output_types = {'path': tf.string,
                          'orig_depth': tf.int16,
                          'orig_spacing': tf.float32,
                          'orig_origin': tf.float32,
                          'orig_direction': tf.float32,
                          'orig_size': tf.int16,
                          'orig_min': tf.float32,
                          'orig_max': tf.float32,
                          'resampled_depth': tf.int16,
                          'resampled_spacing': tf.float32,
                          'resampled_origin': tf.float32,
                          'resampled_direction': tf.float32,
                          'resampled_size': tf.float32,
                          'resampled_min': tf.float32,
                          'resampled_max': tf.float32,
                          'normalization_constant': tf.int16,
                          'data_shape': tf.string
                         }

metadata_output_shapes = {
      'path': (),
      'orig_depth': (),
      'orig_spacing': (3,),
      'orig_origin': (3,),
      'orig_direction': (9,),
      'orig_size': (3,),
      'orig_min': (),
      'orig_max': (),
      'resampled_depth': (),
      'resampled_spacing': (3,),
      'resampled_origin': (3,),
      'resampled_direction': (9,),
      'resampled_size': (3,),
      'resampled_min': (),
      'resampled_max': (),
      'normalization_constant': (),
      'data_shape': ()
}


# dicom_dataset = tf.data.Dataset.from_generator(partial(get_dicom_iterator, root=dataset_dir, size=size),
#                                                output_types=(array_output_type, metadata_output_types),
#                                                output_shapes=(array_output_shape, metadata_output_shapes))

# for i, (arrays, metadata) in enumerate(get_dicom_iterator(dataset_dir)):
    

#     for k in metadata:
#         print(k)
#         metadata[k] = tf.constant(metadata[k], shape=metadata_output_shapes[k], dtype=metadata_output_types[k])
#         metadata[k].numpy()
      
#     for array in arrays:
#         size = array.shape[-1]
#         output_dir = os.path.join(dataset_dir, f'tfrecords_new_{size}x{size}')
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir, exist_ok=True)

#         tensor = tf.constant(array, dtype=tf.float32)
#         to_tfrecords((tensor, metadata), i, output_dir)


def get_dicom_iterator(root):
    for path in get_dcm_paths(root):
        try:
            array, metadata = read_resample_resize_dcm(path)
        except RuntimeError as e:
            print(e)
            print("Continuing...")
            continue
        else:
            yield array, metadata

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def to_tfrecords(tensor_metadata, counter, output_dir):
    # write records to a tfrecords file
    writer = tf.io.TFRecordWriter(os.path.join(output_dir, f'{counter:04}.tfrecord'))

    tensor = tensor_metadata[0]
    metadata = tensor_metadata[1]

    # Loop through all the features you want to write
    features = {}
    features['image'] = tf.train.Feature(float_list=tf.train.FloatList(value=tensor.numpy().flatten()))
    for k in metadata:
        value = metadata[k].numpy()

        if metadata[k].dtype == tf.string:
            feature = _bytes_feature(value)
        elif metadata[k].dtype == tf.float32:
            feature = _float_feature(value.flatten())

        elif metadata[k].dtype == tf.int16:
            feature = _int64_feature(value.flatten())
        else:
            raise NotImplementedError(f"Unsupported dtype {metadata[k].dtype}")

        features[k] = feature

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=features))

    # Serialize the example to a string
    serialized = example.SerializeToString()

    # write the serialized objec to the disk
    writer.write(serialized)
    writer.close()


dataset_dir = '/project/davidr/lidc_idri/'

array_output_type = tf.float32

metadata_output_types = {'path': tf.string,
                          'orig_depth': tf.int16,
                          'orig_spacing': tf.float32,
                          'orig_origin': tf.float32,
                          'orig_direction': tf.float32,
                          'orig_size': tf.int16,
                          'orig_min': tf.float32,
                          'orig_max': tf.float32,
                          'resampled_depth': tf.int16,
                          'resampled_spacing': tf.float32,
                          'resampled_origin': tf.float32,
                          'resampled_direction': tf.float32,
                          'resampled_size': tf.float32,
                          'resampled_min': tf.float32,
                          'resampled_max': tf.float32,
                          'normalization_constant': tf.int16,
                          'data_shape': tf.string
                         }

metadata_output_shapes = {
      'path': (),
      'orig_depth': (),
      'orig_spacing': (3,),
      'orig_origin': (3,),
      'orig_direction': (9,),
      'orig_size': (3,),
      'orig_min': (),
      'orig_max': (),
      'resampled_depth': (),
      'resampled_spacing': (3,),
      'resampled_origin': (3,),
      'resampled_direction': (9,),
      'resampled_size': (3,),
      'resampled_min': (),
      'resampled_max': (),
      'normalization_constant': (),
      'data_shape': ()
}



def apply_fn(arrays, metadata, i):
    for k in metadata:
        metadata[k] = tf.constant(metadata[k], shape=metadata_output_shapes[k], dtype=metadata_output_types[k])
      
    for array in arrays:
        size = array.shape[-1]
        output_dir = os.path.join(dataset_dir, f'tfrecords_new_lanczos_{size}x{size}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tensor = tf.constant(array, dtype=tf.float32)
        to_tfrecords((tensor, metadata), i, output_dir)


# p = Pool()
total = len(os.listdir(dataset_dir))
for i, (arrays, metadata) in enumerate(get_dicom_iterator(dataset_dir)):
    # p.apply_async(apply_fn, args=data)
# p.map_async(apply_fn, get_dicom_iterator(dataset_dir))
    apply_fn(arrays, metadata, i)

    print(f"{i} / {total}")
