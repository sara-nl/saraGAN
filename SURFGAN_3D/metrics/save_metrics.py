import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd
from mpi4py import MPI
import time

from metrics import (calculate_fid_given_batch_volumes, get_swd_for_volumes,
                     get_normalized_root_mse, get_mean_squared_error, get_psnr, get_ssim)

def add_to_metric_summary(metric_name, metric_value, summary_metrics, sess):
    metric_tensor = tf.get_variable(metric_name, dtype = tf.float32, trainable=False, initializer = np.float32(metric_value))
    init_metric = tf.initializers.variables([metric_tensor])
    update_metric = metric_tensor.assign(metric_value)
    sess.run([init_metric, update_metric])
    summary_metrics.append(tf.summary.scalar(metric_name, metric_tensor))

def save_metrics(writer, sess, npy_data, gen_sample, batch_size, global_size, global_step, imagesize_xy, horovod, compute_metrics, num_metric_samples, verbose):
    """
    Saves metrics to a tf.summary

    Parameters:
    -----------
    writer : tf.summary.FileWriter
        The summary filewriter that will store the metrics
    sess : tf.session
        The TensorFlow session with which to compute the metrics
    npy_data : array
        An array of filenames of input samples
    gen_sample : array
        A generated image, as returned py generator().
    batch_size : int
        The (local) batch size
    global_size : int
        The number of workers
    global_step : int
        Global step counter (equals local step counter times number of workers)
    imagesize_xy : int
        Dimension of the image in x,y directions. Needed because swds and ssims require minimum dimensions to be computed
    horovod : bool
        Is horovod used (i.e. was --horovod passed as argument to main)? If true, metrics are averaged over all workers using MPI_allreduce.
    compute_metrics: dictionary of bool
        Uses the keys 'compute_<metric>' with <metric> being swds, ssims, FID, psnrs, mses, nrmses and specifies booleans for whether or not that metric should be computed
    num_metric_samples : int
        Number of batches to compute the metrics on. Metrics are averaged over these batches. Limit is set on the global number of processed batches if Horovod is used.
    verbose : bool
        Should verbose output be printed? (Typically only for rank 0 if Horovod is used)

    Returns:
    --------
    Dictionary of metrics. This dictionary can contains keys swd, ssim, FID, psnr, mse, nrmse. Key for <metric> will only exist if the equivalent key in compute_metrics was true.
    """

    # Initialize metrics dictionary:
    metrics = {}

    # To check timings for metric calculation. Hardcoded, because generally only needed for development...
    report_timings = False

    # Calculate metrics.
    # I'm guessing we only calculate swds and ssims for larger images because they don't make sense on too small images...
    compute_metrics['compute_swds']: bool = (imagesize_xy >= 16 and compute_metrics['compute_swds'])
    compute_metrics['compute_ssims']: bool = (min(npy_data.shape[1:]) >= 16 and compute_metrics['compute_ssims'])
    
    fids_local = []
    swds_local = []
    psnrs_local = []
    mses_local = []
    nrmses_local = []
    ssims_local = []

    counter = 0
    while True:
        if horovod:
            start_loc = counter + hvd.rank() * batch_size
        else:
            start_loc = 0

        real_batch = np.stack([np.load(npy_data[i]) for i in range(start_loc, start_loc + batch_size)])
        real_batch = real_batch[:, np.newaxis, ...].astype(np.float32) / 1024 - 1
        # real_batch = real_batch[:, np.newaxis, ...].astype(np.float32)
        fake_batch = sess.run(gen_sample).astype(np.float32)

        # Turn fake batch into HUs and clip to training range.
        fake_batch = (np.clip(fake_batch, -1, 2) * 1024).astype(np.int16)

#        if verbose:
#            print('Real shape', real_batch.shape)
#            print('Fake shape', real_batch.shape)
#            print('real min, max', real_batch.min(), real_batch.max())
#            print('fake min, max', fake_batch.min(), fake_batch.max())

        if compute_metrics['compute_FID']:
            start = time.time()
            fids_local.append(calculate_fid_given_batch_volumes(real_batch, fake_batch, sess))
            end = time.time()
            if report_timings:
                print("fids took %s" % (end-start))

        if compute_metrics['compute_swds']:
            start = time.time()
            swds = get_swd_for_volumes(real_batch, fake_batch)
            swds_local.append(swds)
            end = time.time()
            if report_timings:
                print("swds took %s" % (end-start))

        if compute_metrics['compute_psnrs']:
            start = time.time()
            psnr = get_psnr(real_batch, fake_batch)
            psnrs_local.append(psnr)
            end = time.time()
            if report_timings:
                print("psnrs took %s" % (end-start))

        if compute_metrics['compute_ssims']:
            start = time.time()
            ssim = get_ssim(real_batch, fake_batch)
            ssims_local.append(ssim)
            end = time.time()
            if report_timings:
                print("ssims took %s" % (end-start))

        if compute_metrics['compute_mses']:
            start = time.time()
            mse = get_mean_squared_error(real_batch, fake_batch)
            mses_local.append(mse)
            end = time.time()
            if report_timings:
                print("mses took %s" % (end-start))
            
        if compute_metrics['compute_nrmses']:
            start = time.time()
            nrmse = get_normalized_root_mse(real_batch, fake_batch)
            nrmses_local.append(nrmse)
            end = time.time()
            if report_timings:
                print("nrmses took %s" % (end-start))

        if horovod:
            counter = counter + global_size * batch_size
        else:
            counter += batch_size

        if counter >= num_metric_samples:
            break

    with tf.variable_scope('metrics', reuse=tf.AUTO_REUSE):

        # Collect metrics in a single tf.summary
        summary_metrics = []

        if compute_metrics['compute_FID']:
            fid_local = np.mean(fids_local)
            if horovod:
                fid = MPI.COMM_WORLD.allreduce(fid_local, op=MPI.SUM) / hvd.size()
            else:
                fid = fid_local
            metrics['FID'] = fid
            if verbose:
                # print(f"FID: {fid:.4f}")
                add_to_metric_summary('fid', fid, summary_metrics, sess)

        if compute_metrics['compute_psnrs']:
            psnr_local = np.mean(psnrs_local)
            if horovod:
                psnr = MPI.COMM_WORLD.allreduce(psnr_local, op=MPI.SUM) / hvd.size()
            else:
                psnr = psnr_local
            metrics['psnr'] = psnr
            if verbose:
                print(f"PSNR: {psnr:.4f}")
                add_to_metric_summary('PSNR', psnr, summary_metrics, sess)

        if compute_metrics['compute_ssims']:
            ssim_local = np.mean(ssims_local)
            if horovod:
                ssim = MPI.COMM_WORLD.allreduce(ssim_local, op=MPI.SUM) / hvd.size()
            else:
                ssim = ssim_local
            metrics['ssim'] = ssim
            if verbose:
                print(f"SSIM: {ssim}")
                add_to_metric_summary('ssim', ssim, summary_metrics, sess)

        if compute_metrics['compute_mses']:
            mse_local = np.mean(mses_local)
            if horovod:
                mse = MPI.COMM_WORLD.allreduce(mse_local, op=MPI.SUM) / hvd.size()
            else:
                mse = mse_local
            metrics['mse'] = mse
            if verbose:
                print(f"MSE: {mse:.4f}")
                add_to_metric_summary('MSE', mse, summary_metrics, sess)

        if compute_metrics['compute_nrmses']:
            nrmse_local = np.mean(nrmses_local)
            if horovod:
                nrmse = MPI.COMM_WORLD.allreduce(nrmse_local, op=MPI.SUM) / hvd.size()
            else:
                nrmse = nrmse_local
            metrics['nrmse'] = nrmse
            if verbose:
                print(f"Normalized Root MSE: {nrmse:.4f}")
                add_to_metric_summary('NRMSE', nrmse, summary_metrics, sess)

        if compute_metrics['compute_swds']:
            swds_local = np.array(swds_local)
            # Average over batches
            swds_local = swds_local.mean(axis=0)
            if horovod:
                swds = MPI.COMM_WORLD.allreduce(swds_local, op=MPI.SUM) / hvd.size()
            else:
                swds = swds_local
            metrics['swd'] = swds
            if verbose:
                print(f"SWDS: {swds}")
                for i in range(len(swds))[:-1]:
                    lod = 16 * 2 ** i
                    add_to_metric_summary(f'swd_{lod}', swds[i], summary_metrics, sess)
                add_to_metric_summary(f'swd_mean', swds[-1], summary_metrics, sess)


        # Finally, write the full summary
        if len(summary_metrics) > 0:
            try:
                summary_metrics = tf.get_default_graph().get_tensor_by_name("metrics/summary_metrics/summary_metrics:0")
            except:
                summary_metrics = tf.summary.merge(summary_metrics, name = "summary_metrics")

            summary_met = sess.run(summary_metrics)
            writer.add_summary(summary_met, global_step)

        return metrics