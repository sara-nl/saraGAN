import cv2
import numpy as np
import glob
import argparse
from tqdm import tqdm
from PIL import Image
import os
import subprocess


def generator(globstr, max_height, max_width, resize_height, resize_width):

    cur_width = 0
    for filename in tqdm(sorted(glob.glob(globstr))):
        img = cv2.imread(filename)

        try:
            height, width, channels = img.shape
        except:
            print(f"Couldn't get shape for {filename}")
            continue

        if width != width:
            #         print(img.shape)
            pass

        if height < max_height:
            factor = max_height // height

            if cur_width != width:
                cur_width = width

            img = cv2.resize(img, (img.shape[1] * factor, img.shape[0] * factor), interpolation=cv2.INTER_NEAREST)
            img = np.pad(img, ((0, 0), (0, max_width - img.shape[1]), (0, 0)), mode='constant')
            assert img.shape == (max_height, max_width, 3)
        yield(cv2.resize(img, (resize_height, resize_width)))


def make_vid(args):
    max_width = int(args.max_width)
    max_height = int(args.max_height)
    os.makedirs('./video_images/')

#     for filename in tqdm(sorted(glob.glob(globstr))):
#         img = cv2.imread(filename)
#
#         try:
#             height, width, channels = img.shape
#         except:
#             print(f"Couldn't get shape for {filename}")
#             continue
#
#         if width != width:
#             #         print(img.shape)
#             pass
#
#         if height < max_height:
#             factor = max_height // height
#
#             if cur_width != width:
#                 cur_width = width
#
#             img = cv2.resize(img, (img.shape[1] * factor, img.shape[0] * factor), interpolation=cv2.INTER_NEAREST)
#             img = np.pad(img, ((0, 0), (0, max_width - img.shape[1]), (0, 0)), mode='constant')
#             assert img.shape == (max_height, max_width, 3)
#         img_array.append(img)
#
#     for img in img_array:
#         assert img.shape == (max_width, max_height, 3), img.shape

    size = (max_height, max_width)
    fps, duration = 24, 100

    # out = cv2.VideoWriter(savefile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, size)

    for i, img in enumerate(generator(args.globstr, max_height, max_width, args.resize_width, args.resize_height)):
        im = Image.fromarray(img, 'RGB')
        im.save("./video_images/%07d.jpg" % i)


   # subprocess.call(
   #      ["ffmpeg", "-y", "-r", str(fps), "-i", "%07d.jpg", "-vcodec", "mpeg4", "-qscale", "5", "-r", str(fps),
   #       "video.avi"])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_width', type=int)
    parser.add_argument('--max_height', type=int)
    parser.add_argument('--resize_height', type=int)
    parser.add_argument('--resize_width', type=int)
    parser.add_argument('--globstr', type=str)
    parser.add_argument('--savefile', type=str)
    args = parser.parse_args()

    make_vid(args)

