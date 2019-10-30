import argparse
import numpy as np
from network_dict import Generator, Discriminator
import datetime
import torch
import os
import SimpleITK as sitk
import time
from glob import glob
# from .image_readers import resample_sitk_image


def write_image(data, filepath, compression=True, metadata=None, resample=False):
    """

    Parameters
    ----------
    data
    filepath : either a path or folder.
        If this is a folder, the output will be written as IMG-<SeriesNumber>-<SliceNumber>.dcm where series number
        is determined by the other files in the folder.
    compression : bool
        Use compression. When the filepath ends with .nii.gz compression is always enabled
    metadata : dict
        Dictionary containing keys such as spacing, direction and origin. If there is a series_description,
        this will be used when writing dicom output
    resample : str
        Will resample the data prior to writing, resampling data should be available in the metadata dictionary.

    Returns
    -------
    None

    TODO: Better catching of SimpleITK errors
    """

    possible_exts = ['nrrd', 'mhd', 'mha', 'nii', 'nii.gz', 'dcm']
    sitk_image = sitk.GetImageFromArray(data)
    # We need to set spacing, otherwise things go wrong.
    sitk_image.SetSpacing(metadata['spacing'])

    if resample and metadata['orig_spacing'] != metadata['spacing']:
        sitk_image, _ = resample_sitk_image(sitk_image, spacing=metadata['orig_spacing'], interpolator=resample)

    if metadata:
        if 'origin' in metadata:
            sitk_image.SetOrigin(metadata['origin'])
        if 'direction' in metadata:
            sitk_image.SetDirection(metadata['direction'])
            
    if any([filepath.endswith(_) for _ in possible_exts]):
        try:
            print(sitk_image.GetSize())
            print(sitk_image.GetSpacing())
            sitk.WriteImage(sitk_image, filepath, True if filepath.endswith('nii.gz') else compression)
        except RuntimeError as e:
            error_str = str(e)
            if error_str.startswith('Exception thrown in SimpleITK WriteImage'):
                if f'Write: Error writing {filepath}' in error_str:
                    raise RuntimeError(f'Cannot write to {filepath}.')
            else:
                raise RuntimeError(e)

    elif os.path.isdir(filepath):
        assert data.ndim == 3, 'For dicom series, only 3D data is supported.'

        series_in_directory = [int(_.split('-')[1]) for _ in glob(os.path.join(filepath, 'IMG-*.dcm'))]
        curr_series = max(series_in_directory)

        modification_time = time.strftime('%H%M%S')
        modification_date = time.strftime('%Y%m%d')
        direction = sitk_image.GetDirection()
        series_tag_values = [
            ('0008|0031', modification_time),  # Series Time
            ('0008|0021', modification_date),  # Series Date
            ('0008|0008', 'DERIVED\\SECONDARY'),  # Image Type
            ('0020|000e', '1.2.826.0.1.3680043.2.1125.' + modification_date + '.1' + modification_time),
            # Series Instance UID
            ('0020|0037', '\\'.join(
                map(str, (
                    direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                    direction[1], direction[4], direction[7])))),
            ('0008|103e', metadata.get('series_description', f'MAPPER OUTPUT {curr_series}'))  # Series Description
        ]
        writer = sitk.ImageFileWriter()
        if compression:
            writer.SetUseCompression()
        # Use the study/series/frame of reference information given in the meta-data dictionary
        writer.KeepOriginalImageUIDOn()
        for idx in range(sitk_image.GetDepth()):
            image_slice = sitk_image[:, :, idx]
            # Set tags specific for series
            for tag, value in series_tag_values:
                image_slice.SetMetaData(tag, value)
            # Set tags specific per slice.
            image_slice.SetMetaData('0008|0012', time.strftime('%Y%m%d'))  # Instance Creation Date
            image_slice.SetMetaData('0008|0013', time.strftime('%H%M%S'))  # Instance Creation Time
            image_slice.SetMetaData('0020|0032', '\\'.join(
                map(str, sitk_image.TransformIndexToPhysicalPoint((0, 0, idx)))))  # Image Position (Patient)
            image_slice.SetMetaData('0020|0013', str(idx))  # Instance Number

            # Write to the output directory and add the extension dcm, to force writing in DICOM format.
            writer.SetFileName(f'IMG-{curr_series:03d}-{idx:03d}.dcm')
            writer.Execute(image_slice)
    else:
        raise ValueError(f'Filename extension has to be one of {possible_exts}')

        
def writeSlices(writer, series_tag_values, new_img, output_dir, i):
    image_slice = new_img[:,:,i]
    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time

    # Setting the type to CT preserves the slice location.
    image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    image_slice.SetMetaData("0020,0013", str(i)) # Instance Number

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(output_dir, str(i)+'.dcm'))
    writer.Execute(image_slice)
        
        
def main(args):
    phase = int(np.log2(args.resolution) - 1)
    num_phases = int(np.log2(args.final_resolution) - 1)
    zdim_base = max(1, args.final_zdim // (2 ** ((num_phases - 1))))
    
    # Get Networks
    generator = Generator(phase, num_phases, args.base_dim, args.latent_dim, (1, zdim_base, 4, 4), nonlinearity='leaky_relu', param=0.3)

    generator.eval()
        
    print(f"Loading weights from {args.run_dir}")
    generator.load_state_dict(torch.load(args.run_dir, map_location=torch.device('cpu')))
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = os.path.join('results', f'{args.resolution}x{args.resolution}', timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    z = torch.randn(args.num_samples, args.latent_dim)
    with torch.no_grad():
        outputs = generator(z, alpha=0)
        
    metadata = {
        'spacing': args.spacing,
        'direction': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }
    
    for i, output in enumerate(outputs):
        output = output.squeeze().cpu().numpy()
        output = (output * args.intercept).astype(np.int16)
        print(output.min(), output.max(), output.shape)
        
        # write_image(output, os.path.join(log_dir, f'{i:02}.dcm'), metadata=metadata)
        
        new_img = sitk.GetImageFromArray(output)
        new_img.SetSpacing(metadata['spacing'])
        
        writer = sitk.ImageFileWriter()
        # Use the study/series/frame of reference information given in the meta-data
        # dictionary and not the automatically generated information from the file IO
        writer.KeepOriginalImageUIDOn()
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")
        # Copy some of the tags and add the relevant tags indicating the change.
        # For the series instance UID (0020|000e), each of the components is a number, cannot start
        # with zero, and separated by a '.' We create a unique series ID using the date and time.
        # tags of interest:
        direction = new_img.GetDirection()
        series_tag_values = [("0008|0031",modification_time), # Series Time
                          ("0008|0021",modification_date), # Series Date
                          ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                          ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                          ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                            direction[1],direction[4],direction[7])))),
                          ("0008|103e", "Created-SimpleITK")] # Series Description
        
        # Write slices to output directory
        output_dir = os.path.join(log_dir, f'{i:02}') 
        os.makedirs(output_dir)
        list(map(lambda j: writeSlices(writer, series_tag_values, new_img, output_dir, j), range(new_img.GetDepth())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('resolution', type=int)
    parser.add_argument('run_dir', type=str)
    parser.add_argument('--intercept', type=int, default=1024)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--final_resolution', type=int, default=512)
    parser.add_argument('--final_zdim', type=int, default=128)
    parser.add_argument('--base_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--spacing', type=tuple, default=(1, 1, 3))
    
    args = parser.parse_args()
    
    main(args)
    