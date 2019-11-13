import cv2
import numpy as np
import glob
import argparse
from tqdm import tqdm


def make_vid(globstr, savefile, max_width, max_height):
    img_array = []
    cur_width = 0

    max_width = int(max_width)
    max_height = int(max_height)

    for filename in tqdm(sorted(glob.glob(globstr))):
        img = cv2.imread(filename)
        height, width, channels = img.shape

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
        img_array.append(img)

    for img in img_array:
        assert img.shape == (max_width, max_height, 3), img.shape

    size = (max_height, max_width)
    out = cv2.VideoWriter(savefile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_width', type=int)
    parser.add_argument('--max_height', type=int)
    parser.add_argument('--globstr', type=str)
    parser.add_argument('--savefile', type=str)
    args = parser.parse_args()

    make_vid(args.globstr, args.savefile, args.max_width, args.max_height)

