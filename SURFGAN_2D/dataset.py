import glob
import numpy as np
from skimage import io, transform
import os
import shutil
import time
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import horovod.tensorflow as hvd


class ImageNetDataset:
    def __init__(self, imagenet_dir, scratch_dir, copy_files, is_correct_phase, num_classes=1):
        super(ImageNetDataset, self).__init__()

        train_folder = os.path.join(imagenet_dir, 'train')
        test_folder = os.path.join(imagenet_dir, 'test')

        classes_train = set(d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d)))
        classes_test = set(d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d)))
        classes_train = classes_test = classes_train.intersection(classes_test)

        classes_train = sorted(list(classes_train))[:num_classes]
        classes_test = sorted(list(classes_test))[:num_classes]

        assert len(classes_train) == len(classes_test) == num_classes

        self.label_to_ix = {label: i for i, label in enumerate(classes_train)}
        self.ix_to_label = {i: label for label, i in self.label_to_ix.items()}

        train_examples = []
        self.train_labels = []

        for label in classes_train:
            for f in glob.glob(os.path.join(train_folder, label) + '/*.JPEG'):
                self.train_labels.append(self.label_to_ix[label])
                train_examples.append(f)

        test_examples = []
        self.test_labels = []

        for label in classes_test:
            for f in glob.glob(os.path.join(test_folder, label) + '/*.JPEG'):
                self.test_labels.append(self.label_to_ix[label])
                test_examples.append(f)

        scratch_dir = os.path.normpath(scratch_dir) if is_correct_phase else imagenet_dir

        self.scratch_files_train = [] if is_correct_phase else train_examples
        self.scratch_files_test = [] if is_correct_phase else test_examples

        if is_correct_phase:
            print("Copying train files to scratch...")
            for f in sorted(train_examples):
                if copy_files:
                    directory = os.path.normpath(scratch_dir + f.rsplit('/', maxsplit=1)[0])
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    shutil.copy(f, os.path.normpath(scratch_dir + f))
                self.scratch_files_train.append(os.path.normpath(scratch_dir + f))
            if copy_files:
                print("All Files Copied")

        if is_correct_phase:
            print("Copying test files to scratch...")
            for f in sorted(test_examples):
                if copy_files:
                    directory = os.path.normpath(scratch_dir + f.rsplit('/', maxsplit=1)[0])
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    shutil.copy(f, os.path.normpath(scratch_dir + f))
                self.scratch_files_test.append(os.path.normpath(scratch_dir + f))

        while not all(os.path.exists(f) for f in self.scratch_files_train):
            print("Waiting...")
            time.sleep(1)

        while not all(os.path.exists(f) for f in self.scratch_files_test):
            print("Waiting...")
            time.sleep(1)

        if is_correct_phase:
            assert all(os.path.isfile(f) for f in self.scratch_files_train)
            assert all(os.path.isfile(f) for f in self.scratch_files_test)

            print(f"Length of train dataset: {len(self.scratch_files_train)}")
            print(f"Length of test dataset: {len(self.scratch_files_test)}")

            assert len(self.scratch_files_train) == len(self.train_labels)
            assert len(self.scratch_files_test) == len(self.test_labels)

        test_image = io.imread(train_examples[0])
        self.shape = test_image.shape
        self.dtype = test_image.dtype

        print(self.shape)
        print(self.dtype)

        del test_image

        self.is_train = True  # TODO: add argument

    def __iter__(self):
        if self.is_train:
            return self.train_labels.__iter__
        else:
            return self.test_labels.__iter__

    def __getitem__(self, idx):
        if self.is_train:
            return self.train_labels[idx]
        else:
            return self.test_labels[idx]

    def __len__(self):
        if self.is_train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)


def imagenet_dataset(imagenet_path, scrath_dir, size, copy_files, is_correct_phase, gpu=False, num_labels=999):
    imagenet_data = ImageNetDataset(imagenet_path, scratch_dir=scrath_dir, copy_files=copy_files, is_correct_phase=is_correct_phase, num_classes=num_labels)

    dataset = tf.data.Dataset.from_tensor_slices((imagenet_data.scratch_files_train, imagenet_data.train_labels))

    def load(path, label):
        # x = np.transpose(transform.resize((io.imread(path.decode()).astype(np.float32) - 127.5) / 127.5, (size, size)), [2, 0, 1])
        y = label
        x = tf.io.read_file(path)
        x = (tf.image.resize(tf.image.decode_jpeg(x, channels=3), [size, size]) - 127.5) / 127.5
        x = tf.transpose(x, perm=[2, 0, 1])
        return x, y

    dataset = dataset.shuffle(len(imagenet_data))

    if gpu:
        parallel_calls = AUTOTUNE
    else:
        parallel_calls = int(os.environ['OMP_NUM_THREADS'])

    # dataset = dataset.map(lambda path, label: tuple(tf.py_func(load, [path, label], [tf.float32, tf.float32])), num_parallel_calls=parallel_calls)
    dataset = dataset.map(load, num_parallel_calls=parallel_calls)
    # dataset = dataset.apply(tf.contrib.data.ignore_errors())
    return dataset


class NumpyPathDataset:
    def __init__(self, npy_dir, scratch_dir, copy_files, is_correct_phase):
        super(NumpyPathDataset, self).__init__()
        self.npy_files = glob.glob(npy_dir + '*.npy')
        print(f"Length of dataset: {len(self.npy_files)}")

        if scratch_dir is not None:
            if scratch_dir[-1] == '/':
                scratch_dir = scratch_dir[:-1]

        self.scratch_dir = os.path.normpath(scratch_dir + npy_dir) if is_correct_phase else npy_dir
        if copy_files and is_correct_phase:
            os.makedirs(self.scratch_dir, exist_ok=True)
            print("Copying files to scratch...")
            for f in self.npy_files:
                # os.path.isdir(self.scratch_dir)
                if not os.path.isfile(os.path.normpath(scratch_dir + f)):
                    shutil.copy(f, os.path.normpath(scratch_dir + f))

        while len(glob.glob(self.scratch_dir + '/*.npy')) < len(self.npy_files):
            time.sleep(1)

        self.scratch_files = glob.glob(self.scratch_dir + '/*.npy')
        assert len(self.scratch_files) == len(self.npy_files)

        test_npy_array = np.load(self.npy_files[0])[np.newaxis, ...]
        self.shape = test_npy_array.shape
        self.dtype = test_npy_array.dtype
        del test_npy_array

    def __iter__(self):
        for path in self.npy_files:
            yield path

    def __getitem__(self, idx):
        return self.npy_files[idx]

    def __len__(self):
        return len(self.npy_files)


if __name__ == '__main__':

    # npy_data = NumpyPathDataset('/lustre4/2/managed_datasets/LIDC-IDRI/npy/average/4x4/', '/scratch-local', copy_files=True,
    #                             is_correct_phase=True)

    imagenet_data = ImageNetDataset('/nfs/managed_datasets/imagenet-full/', scratch_dir='/', copy_files=False, is_correct_phase=True)

    dataset = tf.data.Dataset.from_tensor_slices((imagenet_data.scratch_files_train, imagenet_data.train_labels))


    def load(path, label):
        # x = np.transpose(transform.resize((io.imread(path.decode()).astype(np.float32) - 127.5) / 127.5, (size, size)), [2, 0, 1])
        y = label
        x = tf.io.read_file(path)
        x = (tf.image.resize(tf.image.decode_jpeg(x, channels=3), [32, 32]) - 127.5) / 127.5
        x = tf.transpose(x, perm=[2, 0, 1])
        return x, y

    # Lay out the graph.
    dataset = dataset.shuffle(len(imagenet_data))
    # dataset = dataset.map(lambda path, label: tuple(tf.py_func(load, [path, label], [tf.float32, tf.float32])), num_parallel_calls=int(os.environ['OMP_NUM_THREADS']))
    dataset = dataset.map(load, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(256, drop_remainder=True)
    # dataset = dataset.prefetch(AUTOTUNE)
    # dataset = dataset.repeat()
    dataset = dataset.make_one_shot_iterator()

    real_image_input = dataset.get_next()

    with tf.Session() as sess:
        x, y = sess.run(real_image_input)
        print(x.shape, y.shape, x.min(), x.max())


