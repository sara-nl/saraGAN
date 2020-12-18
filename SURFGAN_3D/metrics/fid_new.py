#!/usr/bin/env python3
''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import os
import tensorflow as tf
from imageio import imread
from scipy import linalg
import pathlib
import warnings
from skimage.transform import resize



class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile(pth, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


# -------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


# -------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    n_images = images.shape[0]
    if batch_size > n_images and verbose:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    n_batches = n_images // batch_size
    pred_arr = np.empty((n_images, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size

        if start + batch_size < n_images:
            end = start + batch_size
        else:
            end = n_images

        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    if verbose:
        print(" done")
    return pred_arr


# -------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# -------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# ------------------
# The following methods are implemented to obtain a batched version of the activations.
# This has the advantage to reduce memory requirements, at the cost of slightly reduced efficiency.
# - Pyrestone
# ------------------


def load_image_batch(files):
    """Convenience method for batch-loading images
    Params:
    -- files    : list of paths to image files. Images need to have same dimensions for all files.
    Returns:
    -- A numpy array of dimensions (num_images,hi, wi, 3) representing the image pixel values.
    """
    return np.array([imread(str(fn)).astype(np.float32) for fn in files])


def get_activations_from_files(files, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    n_imgs = len(files)
    if batch_size > n_imgs:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_imgs
    n_batches = n_imgs // batch_size + 1
    pred_arr = np.empty((n_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        if start + batch_size < n_imgs:
            end = start + batch_size
        else:
            end = n_imgs

        batch = load_image_batch(files[start:end])
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
        del batch  # clean up memory
    if verbose:
        print(" done")
    return pred_arr


def get_activations_from_volume(volume, sess, batch_size=64, verbose=False):
    inception_layer = _get_inception_layer(sess)
    n_imgs = len(volume)

    if batch_size > n_imgs:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_imgs
    n_batches = n_imgs // batch_size + 1
    pred_arr = np.empty((n_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        if start + batch_size < n_imgs:
            end = start + batch_size
        else:
            end = n_imgs

        batch = volume[start:end]

        if len(batch) == 0:
            continue

        batch = (((batch + 1024) / 3072) * 255).astype(int) # [0, 255, int]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
        del batch  # clean up memory
    if verbose:
        print(" done")

    return pred_arr


def calculate_activation_statistics_from_volume(volume, sess, batch_size=64, verbose=False):

    act = get_activations_from_volume(volume, sess,batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = os.getenv('TMPDIR', '/tmp')
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    # print("Checking if %s exists..." % model_file)
    if not model_file.exists():
        # If using multiple MPI ranks, each will download its own inception model.
        # To avoid clashes with multiple ranks, we write in a tempdir.
        # This is inefficient, since we'll download and store one copy of the inception model per rank, but it is fool-proof.
        import tempfile
        dirpath = tempfile.mkdtemp(dir=str(inception_path))
        download_filename = pathlib.Path(dirpath) / 'inception-2015-12-05.tgz'
        model_file = pathlib.Path(dirpath) / 'classify_image_graph_def.pb'
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL, filename=str(download_filename))
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
        model_file = pathlib.Path(dirpath) / 'classify_image_graph_def.pb'

    # print("DEBUG: using inception model path: %s" % str(model_file))
    return str(model_file)


def calculate_fid_given_volumes(volume_real, volume_fake, inception_path, sess):
    inception_path = check_or_download_inception(inception_path)
    create_inception_graph(str(inception_path))

    m1, s1 = calculate_activation_statistics_from_volume(volume_real, sess)
    m2, s2 = calculate_activation_statistics_from_volume(volume_fake, sess)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def calculate_fid_given_batch_volumes(volumes_batch_real, volumes_batch_fake, sess, inception_path=None, data_format='NCDHW'):

    # FID calculation only makes sense if the tensors for fakes and real have the same shape
    if volumes_batch_real.shape != volumes_batch_fake.shape:
        raise Exception("ERROR: unequal shape for batches of real images (%s) and fake images (%s)" % (volumes_batch_real.shape, volumes_batch_fake.shape))

    if volumes_batch_real.ndim != 5 or volumes_batch_fake.ndim != 5:

        # For some reason, our real images enter without a channel dimension
        #if volumes_batch_real.ndim == 4:
        #    print("WARNING: volumes_batch_real.shape = %s, expanding dimensions at axis 1" % volumes_batch_real.shape)
        #    volumes_batch_real = np.expand_dims(volumes_batch_real, 1)
            
        raise Exception("ERROR: either volumes_batch_real.ndim (%s) or volumes_batch_fake.ndim (%s) is not equal to 5." % (volumes_batch_real.ndim,  volumes_batch_fake.ndim))

    if data_format == 'NCDHW':

        volumes_batch_real = np.transpose(volumes_batch_real, [0, 2, 3, 4, 1])
        volumes_batch_fake = np.transpose(volumes_batch_fake, [0, 2, 3, 4, 1])

    # If the image has only 1 channel (typically greyscale), repeat three times, since inception assumes RGB input.
    if volumes_batch_real.shape[-1] == 1:
        volumes_batch_real = np.repeat(volumes_batch_real, 3, axis=-1)
        volumes_batch_fake = np.repeat(volumes_batch_fake, 3, axis=-1)

    # Only the first time this function is called should it load the inception graph. Thus, check if it happens to exist already:
    try:
        layername = 'FID_Inception_Net/pool_3:0'
        sess.graph.get_tensor_by_name(layername)
    except:
        inception_path = check_or_download_inception(inception_path)
        create_inception_graph(str(inception_path))

    # Setting batch_size for FID calculation. 
    # In this context, batch_size is the number of z-slices to be processed in a single batch (i.e. NOT the number of 3D samples)
    batch_size = 64
    if volumes_batch_fake.shape[1] < 64:
        # print("Warning: batch_size for FID calculation (%s) is bigger than the number of z-slices per sample (%s). Setting batch size equal to number of z-slices per sample" % ( batch_size, volumes_batch_fake.shape[1]))
        batch_size = volumes_batch_fake.shape[1]
    
    activations_real = []
    activations_fake = []
    for i in range(len(volumes_batch_fake)):
        # print("DEBUG: Getting activations for volumes %i" % i)
        act_real = get_activations_from_volume(volumes_batch_real[i], sess, batch_size = batch_size)
        act_fake = get_activations_from_volume(volumes_batch_fake[i], sess, batch_size = batch_size)

        activations_real.append(act_real)
        activations_fake.append(act_fake)

    activations_real = np.stack(activations_real)
    activations_fake = np.stack(activations_fake)

    fids = []

    # Use freched_classifier_distance from https://github.com/tsc2017/Frechet-Inception-Distance/blob/master/TF1/fid_tpu_tf1.py
    # This uses tensorflow_gan's tfgan.eval.frechec_classifier_distance_from_activations (https://github.com/tensorflow/gan)
    # Since this is in a for loop AND in a function that gets called many times during training,
    # we first try to get the tensors from the default_graph to see if they already exist.
    # If not, they are created. That should only happen the very first time the FID is computed.
    try:
        activations1 = tf.get_default_graph().get_tensor_by_name("activations1:0")
        activations2 = tf.get_default_graph().get_tensor_by_name("activations2:0")
        fcd = tf.get_default_graph().get_tensor_by_name("fid_from_activations:0")
    except:
        import tensorflow_gan as tfgan
        activations1 = tf.compat.v1.placeholder(tf.float32, [None, None], name = 'activations1')
        activations2 = tf.compat.v1.placeholder(tf.float32, [None, None], name = 'activations2')
        fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)
        fcd = tf.identity(fcd, name = 'fid_from_activations')

    # To avoid the cuSolverDN error that happens when a batch_size of 1 is fed to tfgan.eval.frechet_classifier_distance_from_activations:
    # what if I just reshape the activations_real and activations_fake so as to merge the dimensions that represent the CT ([0]), and the individual slice in the CT ([1])?
    # E.g. if activations_real.shape = (2, 20, 2048) (= batch_size, z-slices, activations), we reshape to (40, 2048). 
    # That way, we just compute the FID based on the activations from each z-slice, where we treat the z-slices completely independently
    activations_real = activations_real.reshape(-1, activations_real.shape[-1])
    activations_fake = activations_fake.reshape(-1, activations_fake.shape[-1])

    fids = sess.run(fcd, feed_dict = {activations1: activations_real, activations2: activations_fake})

    # # WARNING: I think this call is causing the problem with "cuSolverDN call failed with status =6" as soon as the batch size goes to 1.
    # # I think for a batch size of one, the activations_real and *_fake have a first dimension of size 1, which might confuse the fcd op.
    # # Maybe in those cases, I should pass activations_real[0, i, ...] and *fake[0, i, ...] instead?
    # print(f"DEBUG: activations_real.shape: {activations_real.shape}")
    # if activations_real.shape[0] == 1:
    #     fid = sess.run(fcd, feed_dict = {activations1: activations_real[0, ...], activations2: activations_fake[0, ...]})
    # else:
    #     for i in range(activations_real.shape[1]):
    #         #print("DEBUG: Computing frechet distance for activation at depth layer %i out of %i" % (i, activations_real.shape[1]))

    #         #import tensorflow_gan as tfgan
    #         #fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations_real[:, i, ...], activations_fake[:, i, ...])
            
    #         fid = sess.run(fcd, feed_dict = {activations1: activations_real[:, i, ...], activations2: activations_fake[:, i, ...]})
    #         print(f"fid.shape: {fid.shape}")
    #         print(f"fid: {fid}")
    #         # m1 = activations_real[:, i, ...].mean(axis=0)
    #         # m2 = activations_fake[:, i, ...].mean(axis=0)

    #         # sigma1 = np.cov(activations_real[:, i, ...], rowvar=False)
    #         # sigma2 = np.cov(activations_fake[:, i, ...], rowvar=False)

    #         # fid = calculate_frechet_distance(m1, sigma1, m2, sigma2)

    #         # print("Computed FID using old function: %s" % fid)

    #         fids.append(fid)

    # fids = np.stack(fids)
    # print(f"fids.shape: {fids.shape}")

    # assert len(fids) == volumes_batch_real.shape[1]

    return fids


if __name__ == "__main__":

        volumes_real = ((np.random.rand(1, 1, 16, 64, 64) * 3072) - 1024).astype(int)
        volumes_fake = ((np.random.rand(1, 1, 16, 64, 64) * 3072) - 1024).astype(int)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            fid_value = calculate_fid_given_batch_volumes(volumes_real, volumes_fake, sess)

        print("FID: ", fid_value)
