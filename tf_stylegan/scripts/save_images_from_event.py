import os
import imageio
import tensorflow as tf
import argparse


def save_images_from_event(fn, tag, output_dir='./', save_every_n=1):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for i, e in enumerate(tf.train.summary_iterator(fn)):
            if i % save_every_n != 0:
                continue
            for j, v in enumerate(e.summary.value):
                print(i, j, v.tag, tag)
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    imageio.imwrite(output_fn, im)
                    count += 1  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--save_every_n', type=int, default=1)
    args = parser.parse_args()
    save_images_from_event(args.summary_file, args.tag, args.output_dir, args.save_every_n)
