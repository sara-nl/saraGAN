import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import ast
import imageio
import SimpleITK as sitk


def parse_tuple(string):
    try:
        s = ast.literal_eval(str(string))
        if type(s) == tuple:
            return s
        return
    except:
        return


def normalize(x, logical_minimum, logical_maximum):
    x = x.astype(np.float32)
    x = np.clip(x, logical_minimum, logical_maximum)
    x = (x - x.min()) / (x.max() - x.min())  # [0, 1]
    assert x.min() >= 0 and x.max() <= 1
    x = (x * 255).astype(np.uint8)
    return x


def make_x_vid(array, out_dir, logical_maximum=2, logical_minimum=-1):

    shape = array.shape
    # logical_maximum = array.max()
    img_array_x = []
    constant_slice_z = shape[0] // 2
    constant_slice_y = shape[2] // 2

    for i in range(shape[1]):
        array_z = array[constant_slice_z, :, :]
        array_z_plot = array_z.copy()
        array_z_plot[i, :] = logical_maximum
        array_z_plot[:, constant_slice_y] = logical_maximum

        array_x = array[:, i, :]

        array_y = array[:, :, constant_slice_y]
        array_y_plot = array_y.copy()
        array_y_plot[:, i] = logical_maximum
        array_y_plot[constant_slice_z, :] = logical_maximum
        target_size = array_z.shape[0]  # Assuming x, and y are the largest dims.
        num_pad = (target_size - shape[0]) // 2
        array_x = np.pad(array_x, ((num_pad, num_pad), (0, 0)), mode='constant', constant_values=logical_minimum)
        assert array_x.shape == (target_size, target_size)
        half_size = target_size // 2
        num_pad = (shape[1] - shape[0]) // 2
        array_y_plot = np.pad(array_y_plot, ((num_pad, num_pad), (0, 0)), mode='constant', constant_values=logical_minimum)
        assert min(array_y_plot.shape) == target_size
        array_y_plot = cv2.resize(array_y_plot, (half_size, half_size))
        assert array_y_plot.shape == (half_size, half_size)

        array_z_plot = cv2.resize(array_z_plot, (half_size, half_size))

        img_array_x.append(np.concatenate([array_x, np.concatenate([array_z_plot, array_y_plot], axis=0)], axis=1))

    vid = normalize(np.stack(img_array_x), logical_minimum, logical_maximum)
    imageio.mimwrite(os.path.join(out_dir, 'x_vid.mp4'), vid, fps=15)
    return vid


def make_y_vid(array, out_dir, logical_maximum=2, logical_minimum=-1):
    shape = array.shape
    img_array_y = []
    constant_slice_z = shape[0] // 2
    constant_slice_y = shape[2] // 2

    for i in range(shape[2]):
        array_z = array[constant_slice_z, :, :]
        array_z_plot = array_z.copy()
        array_z_plot[:, i] = logical_maximum
        array_z_plot[constant_slice_y, :] = logical_maximum

        array_y = array[:, :, i]

        array_x = array[:, constant_slice_y, :]
        array_x_plot = array_x.copy()
        array_x_plot[:, i] = logical_maximum
        array_x_plot[constant_slice_z, :] = logical_maximum

        target_size = array_z.shape[0]  # Assuming x, and y are the largest dims.
        num_pad = (target_size - shape[0]) // 2
        array_y = np.pad(array_y, ((num_pad, num_pad), (0, 0)), mode='constant', constant_values=logical_minimum)
        assert array_y.shape == (target_size, target_size)

        half_size = target_size // 2
        num_pad = (target_size - shape[0]) // 2
        array_x_plot = np.pad(array_x_plot, ((num_pad, num_pad), (0, 0)), mode='constant', constant_values=logical_minimum)
        assert min(array_x_plot.shape) == target_size
        array_x_plot = cv2.resize(array_x_plot, (half_size, half_size))

        array_z_plot = cv2.resize(array_z_plot, (half_size, half_size))

        img_array_y.append(np.concatenate([array_y, np.concatenate([array_z_plot, array_x_plot], axis=0)], axis=1))

    vid = normalize(np.stack(img_array_y), logical_minimum, logical_maximum)

    imageio.mimwrite(os.path.join(out_dir, 'y_vid.mp4'), vid, fps=15)
    return vid


def make_z_vid(array, out_dir, logical_maximum=2, logical_minimum=-1):
    shape = array.shape

    img_array_z = []
    constant_slice_x = shape[1] // 2
    constant_slice_y = shape[2] // 2

    def add_slice(i):
        array_y = array[:, :, constant_slice_y]
        array_y_plot = array_y.copy()
        array_y_plot[i, :] = logical_maximum
        array_y_plot[:, constant_slice_y] = logical_maximum

        array_z = array[i, :, :]

        array_x = array[:, constant_slice_y, :]
        array_x_plot = array_x.copy()
        array_x_plot[i, :] = logical_maximum
        array_x_plot[:, constant_slice_x] = logical_maximum

        target_size = array_z.shape[0]  # Assuming x, and y are the largest dims.
        half_size = target_size // 2
        num_pad = (target_size - shape[0]) // 2

        array_x_plot = np.pad(array_x_plot, ((num_pad, num_pad), (0, 0)), mode='constant', constant_values=logical_minimum)
        array_y_plot = np.pad(array_y_plot, ((num_pad, num_pad), (0, 0)), mode='constant', constant_values=logical_minimum)

        assert min(array_x_plot.shape) == target_size
        assert min(array_y_plot.shape) == target_size

        array_x_plot = cv2.resize(array_x_plot, (half_size, half_size))
        array_y_plot = cv2.resize(array_y_plot, (half_size, half_size))

        img_array_z.append(np.concatenate([array_z, np.concatenate([array_y_plot, array_x_plot], axis=0)], axis=1))

    for i in range(shape[0]):
        add_slice(i)

    for i in reversed(range(shape[0])):
        add_slice(i)

    vid = normalize(np.stack(img_array_z), logical_minimum, logical_maximum)

    imageio.mimwrite(os.path.join(out_dir, 'z_vid.mp4'), vid, fps=15)
    return vid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_file', type=str)
    parser.add_argument('--out_dir', type=str)
    # parser.add_argument('--shape', type=str, default='(64, 256, 256)')
    parser.add_argument('--flip', type=bool, default=True)
    args = parser.parse_args()

    array = np.load(args.npy_file)
    array = np.flip(array, axis=0)

    image = sitk.GetImageFromArray(array)
    image.SetSpacing([1, 1, 3])
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = sitk.sitkLinear
    new_spacing = [1, 1, 1]
    resample.SetOutputSpacing(new_spacing) 
    
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = image.GetSpacing()
    new_size = orig_size*(orig_spacing/np.array(new_spacing))
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    
    array = sitk.GetArrayFromImage(resample.Execute(image))

    # shape = parse_tuple(args.shape)
    print(f"Array shape: {array.shape}. Min: {array.min()}, Max: {array.max()}")
    # print(f"Target shape: {shape}")

    # plt.imshow(array[32], cmap='gray')
    # plt.show()
    # plt.close()

    xvid = make_x_vid(array, args.out_dir)
    yvid = make_y_vid(array, args.out_dir)
    zvid = make_z_vid(array, args.out_dir)

    final_vid = np.concatenate([xvid, yvid, zvid], axis=0)
    imageio.mimwrite(os.path.join(args.out_dir, 'final.mp4'), final_vid, fps=30)

