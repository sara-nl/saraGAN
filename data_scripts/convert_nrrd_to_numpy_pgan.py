import os, glob, re
import nrrd
import numpy as np
from skimage.measure import block_reduce

DEBUG=False
min_array=[]
intercept = -2048
clip_value = 2048


def convert_all(prefix='/projects/0/radioct/14/', saveprefix='/projects/0/radioct/14_pgan/', ext='.nrrd', reduce_fn=np.average):
    header_array = []
    search_string = re.compile(f"{prefix}(.*)")
    #print("Getting all files with glob pattern:")
#    print(glob.glob(f"{prefix}*/*/*{ext}"))

    print(f"Reading files with extension {ext} from {prefix}")
    print(f"Saving to prefix {saveprefix}")

# Some counters for reporting
    counters = {
        "discarded": 0,
        "cropped": 0,
        "padded": 0,
        "zdim": 0,
    }

    for file in glob.glob(f"{prefix}/*/*/*{ext}"):
        print("")
        print(f"Processing: {file}")
        # TESTING: only on the first 100 files
#        if i < 100:
#            i = i+1
        savefilename = search_string.match(file).group(1).replace('/','_')
        # Strip .nrrd extension since we want to save numpy files
        savefilename = os.path.splitext(savefilename)[0]
        savefilename = f'{savefilename}.npy'

        tmp_header = convert_one(file, saveprefix, savefilename, reduce_fn, counters)
        if tmp_header:
            header_array.append(tmp_header)
        print(counters)

    nheaders = len(header_array)
    print(f"Selected #headers: {nheaders}")
    print(f"Discarded: {counters['discarded']}")
    print(f"Cropped: {counters['cropped']}")
    print(f"Padded: {counters['padded']}")
#    dimz=[header['sizes'][2] for header in header_array]
#    print(np.histogram(dimz))
#    print(max(dimz))
#        print(file)

# Check if we want to the scan is 'normal' in terms of dimensions, etc
def keep_scan(header, inputfile):
    # Check x,y dimensions. Should only discard a single one from the current dataset.
    xy_dim = header['sizes'][0:2]
    xy_dim_valid = [512,512]
    if not (xy_dim == xy_dim_valid).all():
        print(f"Warning: skipping conversion of file {inputfile}, because it doesn't have the correct dimensions: {xy_dim} (expected: {xy_dim_valid}")
        return False

    # Discard images larger than 190
    zdim = header['sizes'][2]
    zdim_valid = 190
    if zdim >= 190:
        return False

    # Check x,y spacing, discard if negative because it means one of the axes is inverted. Should only discard a single one from the current dataset
    spacing = header['space directions']
    spacing00 = spacing[0,0]
    spacing11 = spacing[1,1]
    spacing22 = spacing[2,2]
    # 2775 out of 2968 have a spacing00 and spacing11 of either 0.9765625 or 0.976563
    if not (spacing00 == 0.9765625 or spacing00 == 0.976563) :
        print(f"Warning: skipping conversion of file {inputfile}, because it doesn't have the correct x spacing: ({spacing00} instead of 0.9765625 or 0.976563)")
        return False
    if not (spacing11 == 0.9765625 or spacing11 == 0.976563) :
        print(f"Warning: skipping conversion of file {inputfile}, because it doesn't have the correct x spacing: ({spacing11} instead of 0.9765625 or 0.976563)")
        return False
    # 2772 out of 2968 have a spacing22 of 3.0mm. Let's only select those:
    if not spacing22 == 3.0:
        print(f"Warning: skipping conversion of file {inputfile}, because it doesn't have the correct slice thickness ({spacing22} instead of 3.0mm)")
        return False

    # If we reached here, all requirements for the scan are satisfied to be accepted
    return True

def convert_one(inputfile, saveprefix, savefilename, reduce_fn, counters):
#    print(f"Converting file {inputfile}")
    header = nrrd.read_header(inputfile)

    # If we don't want to keep the scan, do nothing
    if not keep_scan(header, inputfile):
        counters['discarded'] = counters['discarded'] + 1
        return False

    # Let's count how many more we loose if we discard above 180
#    if header['sizes'][2] > 190:
#        counters['zdim'] = counters['zdim'] + 1

    # If we are here, scan is accepted, so continue to read the actual data
    ct_array, header = nrrd.read(inputfile)

    # Padding & cropping
    target_dim = [512, 512, 160]

    old_dim = ct_array.shape
    ct_array = pad_to(ct_array, target_dim)
    if old_dim != ct_array.shape:
        counters['padded'] = counters['padded'] + 1
        print(f"Padded from: {old_dim} to {ct_array.shape}")

    old_dim = ct_array.shape
    ct_array = crop_to(ct_array, target_dim)
    if old_dim != ct_array.shape:
        counters['cropped'] = counters['cropped'] + 1
        print(f"Cropped from: {old_dim} to {ct_array.shape}")

    # Reorder dimensions to put zdim first, since this is what SURFGAN3D expects
    ct_array = np.moveaxis(ct_array, -1, 0)

    # Rescale values by the intercept, which SEEMS to be -2048 in these scans...
    if DEBUG:
        minimum = np.minimum(ct_array)
        print(f'Minimum: {minimum}')
        min_array.append(minimum)
    ct_array = ct_array - intercept

    # Loop over all dimensions, starting with the original
    for i in range(0,6):
        # Downsample except first iteration
        if i == 0:
            reduced = ct_array
        else:
            kernel = 2 ** i
            reduced = block_reduce(ct_array, (kernel, kernel, kernel), func=reduce_fn, cval=0)

        # Clip and change datatype to uint16
        reduced = np.clip(reduced, 0, clip_value - intercept)
        reduced = reduced.astype(np.uint16)
    
        # Save original size
        size = reduced.shape[-1]
        npy_dir = os.path.join(saveprefix,'npy', reduce_fn.__name__, f'{size}x{size}')
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)
        
        print(f'Saving {savefilename} at size {reduced.shape})')
        np.save(os.path.join(npy_dir, savefilename), reduced)
    
    return header

def pad_to(data, target_dim, center=[True, True, True]):
    '''Pad numpy array in data to the target dimensions specified in target_dim. By default, the array will receive padding on all sides. If you set center=[False, False, False], the padding will be done on the at the end of each dimension'''
    source_dim = data.shape
    pad_size = [target - current for target, current in zip(target_dim, source_dim)]
    # Initialize:
    pad_min = [0, 0, 0]
    pad_max = [0, 0, 0]
    for i in range(3):
        # Image is already larger than padding in this axis:
        if pad_size[i] <= 0:
            pad_min[i] = 0
            pad_max[i] = 0
        # Padding while centering the image:
        elif center[i]:
            pad_min[i] = np.floor(pad_size[i]/2)
            pad_max[i] = np.ceil(pad_size[i]/2)
        # Padding at the end of the axis:
        else:
            pad_min[i] = 0
            pad_max[i] = pad_size[i]
    # Restructure and cast to int, as required by np.pad
    pad_list = [(int(pmin), int(pmax)) for pmin, pmax in zip(pad_min, pad_max)]
    if DEBUG:
        print(f"Padding image from {source_dim} to {target_dim} using padding list: {pad_list}")
    # We use the intercept as padding value, that makes the most sense since then, after rescaling, it will be '0'
    padded_data = np.pad(data, pad_list, mode='constant', constant_values=intercept)

    # Sanity check:
    assert all( [ tdim <= pdim for tdim, pdim in zip(target_dim, padded_data.shape) ] )

    return padded_data

def crop_to(data, target_dim, center=[True, True, True]):
    '''Crop numpy array in data to the target dimensions specified in target_dim. By default, the image will be cropped from all sides. If you set center=[False, False, False], cropping will be done from the end of each dimension'''
    source_dim = data.shape
    start = [ ((source // 2) - (target // 2)) for source,target in zip(source_dim, target_dim)]
    end = [start_i + target_i for start_i,target_i in zip(start, target_dim)]
    
    # Doing the actual crop:
    cropped_data = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    if DEBUG:
        print(f"Cropping starts: {start}")
        print(f"Cropping ends: {end}")
        print(f"Size after crop: {cropped_data.shape}")

    assert all( [ tdim >= cdim for tdim, cdim in zip(target_dim, cropped_data.shape) ] )

    return cropped_data

convert_all()

if DEBUG:
    print('Minimum Histogram:')
    print(np.histogram(min_array))
