import tensorflow as tf
import os
from functools import partial
import SimpleITK as sitk
import numpy as np
import skimage
import argparse
from skimage.measure import block_reduce
import pandas as pd
from multiprocessing import Pool
import time

def get_dcm_paths(root):
    for i, (directory, subdirectories, files) in enumerate(os.walk(root)):
        print(i)
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

#### Get statistics

root = '/project/davidr/lidc_idri/'

def track_job(job, update_interval=3):
    while job._number_left > 0:
        print("Tasks remaining = {0}".format(
        job._number_left * job._chunksize))
        time.sleep(update_interval)



def map_fn(path):
    image = read_dcm_series(path)    
    metadata = {}
    
    metadata['path'] = path
    metadata['orig_depth'] = image.GetDepth()
    metadata['orig_spacing'] = tuple(image.GetSpacing())
    metadata['orig_origin'] = tuple(image.GetOrigin())
    metadata['orig_direction'] = tuple(image.GetDirection())
    metadata['orig_size'] = tuple(image.GetSize())
     
    array = sitk.GetArrayFromImage(image)
    metadata['orig_min'] = array.min()
    metadata['orig_max'] = array.max()
    
    metadata['orig_mean'] = array.mean()
    metadata['orig_std'] = array.std()
    metadata['orig_median'] = np.median(array)
    
    return metadata

p = Pool()
res = p.map_async(map_fn, get_dcm_paths(root))
track_job(res)

metadatas = res.get()
metadata_df = pd.DataFrame(metadatas)
metadata_df.to_csv('metadata.csv', index=None)