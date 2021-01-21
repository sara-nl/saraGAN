# pylint: disable=import-error
import argparse
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import time
import random
import optuna

from utils import get_verbosity, print_study_summary
from mpi4py import MPI
import os
import psutil

from tensorflow.data.experimental import AUTOTUNE
import nvgpu

from optuna_objective import optuna_objective

# For TensorBoard Debugger:
from tensorflow.python import debug as tf_debug

def main(args, config):

    verbose = get_verbosity(args.horovod, args.optuna_distributed)

    # See how much output we can get...
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Raised errors that should be caught, but trials should just continue (errors are e.g. thrown when OOM)
    catchErrorsInTrials = (tf.errors.UnknownError, tf.errors.InternalError)

    # We support several types of runs:
    # Load hyperparameters from the best trial in an optuna database, do a (potentially data-parallel) convergence run
    run_from_best_trial = (args.optuna_use_best_trial and (args.optuna_storage is not None))
    # Optimize hyperparameters. If multiple MPI workers are launched, each worker runs a single Optuna Trial
    hyperparam_opt_inter_trial = args.optuna_distributed and not run_from_best_trial
    # Optimize hyperparameters. If multiple MPI workers are launched, workers work together on a single trial. You can start such runs multiple times to also parallelize over multiple trials.
    hyperparam_opt_intra_trial = (args.optuna_storage is not None) and (args.optuna_study_name is not None) and not hyperparam_opt_inter_trial
    # Normal run, no hyperparameter tuning. Do a (potentially data-parallel) convergence run
    normal_run = (not run_from_best_trial) and (not hyperparam_opt_inter_trial) and (not hyperparam_opt_intra_trial)

    if normal_run:
        if verbose:
            print("Performing single training run (no hyperparameter tuning)")
        optuna_objective(None, args, config)

    elif run_from_best_trial:
        if verbose:
            print("Performing single training run using hyperparameters previously optimized by Optuna")

        if args.optuna_study_name is None:
            study_name = optuna.study.get_all_study_summaries(args.optuna_storage)[0].study_name
        else:
            study_name = args.optuna_study_name

        if verbose:
            print("Restoring best trial:")
            print(f"    Study name: {study_name}")
            print(f"    Database: {args.optuna_storage}")
        study = optuna.load_study(study_name = study_name, storage = args.optuna_storage)

        # Start a full training with the best_trial parameters that were obtained previously:
        if verbose:
            print("Running a single training with the following fixed trial parameters:")
            print(study.best_trial)
        optuna_objective(study.best_trial, args, config)

    elif hyperparam_opt_inter_trial:
        if verbose:
            print("Performing hyperparameter optimization run with Optuna. If MPI is used, each rank runs a single trial.")

        if args.horovod:
            print("You can either distribute optuna trials over MPI workers, or run a single trial in dataparallel fashion. To do both, please pre-create an optuna database and launch multiple runs to parallelize trials, while using MPI to parallize WITHIN a trial.")
            raise NotImplementedError()

        # Automatically generate a study name and storage, if they were not passed on command line
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        if args.optuna_study_name is None:
            study_name = f"optuna_{timestamp}"
        else:
            study_name = args.optuna_study_name
        if args.optuna_storage is None:
            if args.logdir is not None:
                storage_sqlite=f'sqlite:///{args.logdir}/optuna.db'
            else:
                storage_sqlite = f'sqlite:///optuna_{timestamp}.db'
        else:
            storage_sqlite = args.optuna_storage

        # Only worker with rank 0 should create a study:
        study = None
        if hvd.rank() == 0:
            print("Storing SQlite database for optuna at %s" %storage_sqlite)
            study = optuna.create_study(direction = "minimize", study_name = study_name, storage = storage_sqlite)
    
        # Call a barrier to make sure the study has been created before the other workers load it
        MPI.COMM_WORLD.Barrier()

        # Then, make all other workers load the study
        if hvd.rank() != 0:
            study = optuna.load_study(study_name = study_name, storage = storage_sqlite)

        ntrials = np.ceil(args.optuna_ntrials/hvd.size())

        study.optimize(lambda trial: optuna_objective(trial, args, config), n_trials = ntrials, catch = catchErrorsInTrials)

        print_study_summary(study)

    elif hyperparam_opt_intra_trial:
        if verbose:
            print("Performing hyperparameter optimization run with Optuna. If MPI is used, it is used to perform data-parallel training")

        study = optuna.load_study(study_name = args.optuna_study_name, storage = args.optuna_storage)
        ntrials = args.optuna_ntrials
        # If using horovod for intra-trial parallelization (data-parallel), only the first worker calls study.optimize. The others call optuna_objective directly.
        if args.horovod and (hvd.rank() != 0):
            # Execute same number of trials 
            for i in range(ntrials):
                # Get trial and args
                trial = None
                args = None
                trial = MPI.COMM_WORLD.bcast(trial, root = 0)
                args = MPI.COMM_WORLD.bcast(args, root = 0)
                print(f'Worker: {hvd.rank()} received trial {trial} and arguments {args}')
                optuna_objective(trial, args, config)
        else:
            study.optimize(lambda trial: optuna_objective(trial, args, config), n_trials = ntrials, catch = catchErrorsInTrials)

        print_study_summary(study)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str)
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--start_shape', type=str, default=None, required=True, help="Shape of the data at phase 0, '(c, z, y, x)', e.g. '(1, 5, 16, 16)'")
    parser.add_argument('--final_shape', type=str, default=None, required=True, help="'(c, z, y, x)', e.g. '(1, 64, 128, 128)'")
    parser.add_argument('--starting_phase', type=int, default=None, required=True)
    parser.add_argument('--ending_phase', type=int, default=None, required=True)
    parser.add_argument('--scratch_path', type=str, default=None, required=True)
    parser.add_argument('--base_batch_size', type=int, default=None, help='batch size used in phase 1')
    parser.add_argument('--max_global_batch_size', type=int, default=256)
    parser.add_argument('--mixing_nimg', type=int, default=2 ** 19)
    parser.add_argument('--stabilizing_nimg', type=int, default=2 ** 19)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horovod', default=False, action='store_true')
    parser.add_argument('--checkpoint_every_nsteps', default=20000, type=int, help="Checkpoint files are saved every time the globally processed image counter is (approximately) a multiple of this number. Technically, the counter needs to satisfy: counter % checkpoint_every_nsteps < global_batch_size.")
    parser.add_argument('--logdir', default=None, type=str, help="Allows one to specify the log directory. The default is to store logs and checkpoints in the <repository_root>/runs/<network_architecture>/<datetime_stamp>. You may want to override from the batch script so you can store additional logs in the same directory, e.g. the SLURM output file, job script, etc")
    parser.add_argument('--continue_path', default=None, type=str)
    parser.add_argument('--starting_alpha', default=1, type=float)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--num_inter_ops', default=4, type=int)
    parser.add_argument('--num_labels', default=None, type=int)
    parser.add_argument('--validation_fraction', default=0.1, type=float, required=False, help="Fraction of the dataset that will be set aside for validation. Metrics are computed on the validation data set at regular intervals during training (stored in the tensorflow events file. Also see metrics_every_nsteps) and at the end of a phase (reported on the command line. Also see compute_metrics_validation). Optuna optimizes hyperparameters based on the FID computed on the validation data set.")
    parser.add_argument('--test_fraction', default=0.1, type=float, required=False, help="Fraction of the dataset that will be set aside for testing. Metrics can be computed on the test data set at the end of each phase (see compute_metrics_test).")

    # Architecture
    parser.add_argument('--latent_dim', type=int, default=None, required=True)
    parser.add_argument('--first_conv_nfilters', type=int, default=None, required=True, help='Number of filters in the first convolutional layer. Since it is densely connected to the latent space, the number of connections can increase rapidly, hence it can be set separately from the other filter counts deeper in the network')
    parser.add_argument('--network_size', default=None, choices=['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl'], required=True)
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.2)

    # Learning rate
    parser.add_argument('--g_lr', type=float, default=None)
    parser.add_argument('--d_lr', type=float, default=None)
    parser.add_argument('--g_lr_increase', type=str, choices=[None, 'linear', 'exponential'], default=None, help='Defines if the learning rate should gradually increase to g_lr at the start of each phase, and if so, if this should happen linearly or exponentially. For exponential increase, the starting value is 1% of g_lr')
    parser.add_argument('--g_lr_decrease', type=str, choices=[None, 'linear', 'exponential'], default=None, help='Defines if the learning rate should gradually decrease from g_lr at the end of each phase, and if so, if this should happen linearly or exponentially. For exponential decrease, the final value is 1% of g_lr')
    parser.add_argument('--d_lr_increase', type=str, choices=[None, 'linear', 'exponential'], default=None, help='Defines if the learning rate should gradually increase to d_lr at the start of each phase, and if so, if this should happen linearly or exponentially. For exponential increase, the starting value is 1% of d_lr')
    parser.add_argument('--d_lr_decrease', type=str, choices=[None, 'linear', 'exponential'], default=None, help='Defines if the learning rate should gradually decrease from d_lr at the end of each phase, and if so, if this should happen linearly or exponentially. For exponential decrease, the final value is 1% of d_lr')
    parser.add_argument('--g_lr_rise_niter', type=int, default=None, help='If a learning rate schedule with a gradual increase in the beginning of a phase is defined for the generator, this number defines within how many iterations the maximum is reached.')
    parser.add_argument('--g_lr_decay_niter', type=int, default=None, help='If a learning rate schedule with a gradual decrease at the end of a phase is defined for the generator, this defines within how many iterations the minimum is reached.')
    parser.add_argument('--d_lr_rise_niter', type=int, default=None, help='If a learning rate schedule with a gradual increase in the beginning of a phase is defined for the discriminator, this number defines within how many iterations the maximum is reached.')
    parser.add_argument('--d_lr_decay_niter', type=int, default=None, help='If a learning rate schedule with a gradual decrease at the end of a phase is defined for the discriminator, this defines within how many iterations the minimum is reached.')
    parser.add_argument('--d_scaling', default='none', choices=['linear', 'sqrt', 'none'],
                        help='How to scale discriminator learning rate with horovod size.')
    parser.add_argument('--g_scaling', default='none', choices=['linear', 'sqrt', 'none'],
                        help='How to scale generator learning rate with horovod size.')

    # Loss & optimization
    parser.add_argument('--loss_fn', default='logistic', choices=['logistic', 'wgan'])
    parser.add_argument('--gp_weight', type=float, default=1)
    parser.add_argument('--g_clipping', default=False, type=bool)
    parser.add_argument('--d_clipping', default=False, type=bool)
    parser.add_argument('--optim_strategy', default='simultaneous', choices=['simultaneous', 'alternate'])
    parser.add_argument('--beta1', type=float, default=0)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--use_adasum', default=False, action='store_true')
    parser.add_argument('--ema_beta', type=float, default=0.99)
    parser.add_argument('--noise_stddev', default=None, type=float, required=True, help="Normally distributed noise is added to the inputs before training ('instance noise', see e.g. https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/). This argument specifies the standard deviation of that normal distribution, and thus the magnitude of that noise. Adding noise that is of the same order as the real noise in your image likely has the best effect.")

    # Not sure if these do anything anymore...
    parser.add_argument('--g_annealing', default=1,
                        type=float, help='generator annealing rate, 1 -> no annealing.')
    parser.add_argument('--d_annealing', default=1,
                        type=float, help='discriminator annealing rate, 1 -> no annealing.')

    # Metrics
    parser.add_argument('--calc_metrics', default=False, action='store_true')
    parser.add_argument('--compute_metrics_train', default=False, action='store_true', help="If defined, all metrics will be computed on the full training data set at the end of a resolution step / 'phase' (not recommended, very time consuming).")
    parser.add_argument('--disable_compute_metrics_validation', dest = 'compute_metrics_validation', default=True, action='store_false', help="If defined, all metrics will be computed on the full validation data set at the end of a resolution step / 'phase' (recommended, but time consuming).")
    parser.add_argument('--disable_compute_metrics_test', dest = 'compute_metrics_test', default=True, action='store_false', help="If defined, all metrics will be computed on the full test data set at the end of a resolution step / 'phase' (recommended, but time consuming).")
    parser.add_argument('--summary_small_every_nsteps', default=32, type=int, help="Summaries are saved every time the locally processsed image counter is a multiple of this number")
    parser.add_argument('--summary_large_every_nsteps', default=64, type=int, help="Large summaries such as images are saved every time the locally processed image counter is a multiple of this number")
    parser.add_argument('--num_metric_samples', type=int, default=None, help="Number of samples used to compute the metrics are computed. A higher number of samples will be more accurate (show less variation in metrics between iterations), but take more time to compute.")
    parser.add_argument('--metrics_every_nsteps', default=128, type=int, help="Metrics are computed every time the locally processed image counter is a multiple of this number")
    parser.add_argument('--metrics_batch_size', default=16, type=int, help="Batch size used for computing the metrics. Note that metrics computation is often a lot lighter than training, so memory limits are less like to be a problem.")
    parser.add_argument('--compute_FID', default=False, action='store_true', help="Whether to compute the Frechet Inception Distance (frequency determined by metrics_every_nsteps)")
    parser.add_argument('--compute_swds', default=False, action='store_true', help="Whether to compute the Sliced Wasserstein Distance (frequency determined by metrics_every_nsteps)")
    parser.add_argument('--compute_ssims', default=False, action='store_true', help="Whether to compute the Structural Similarity (frequency determined by metrics_every_nsteps)")
    parser.add_argument('--compute_psnrs', default=False, action='store_true', help="Whether to compute the peak signal to noise ratio (frequency determined by metrics_every_nsteps). Not very meaningfull for GANs...")
    parser.add_argument('--compute_mses', default=False, action='store_true', help="Whether to compute the mean squared error (frequency determined by metrics_every_nsteps). Not very meaningfull for GANs...")
    parser.add_argument('--compute_nrmses', default=False, action='store_true', help="Whether to compute the normalized mean squared error (frequency determined by metrics_every_nsteps). Not very meaningfull for GANs...")

    # Optuna
    parser.add_argument('--optuna_distributed', default=False, action='store_true', help="Pass this argument if you want to run optuna in distributed fashion. Run should be started as an mpi program (i.e. launching with mpirun or srun). Each MPI rank will work on its own Optuna trial. Do NOT combine with --horovod: parallelization happens at the trial level, it should NOT also be done within trials.")
    parser.add_argument('--optuna_ntrials', default=100, type=int, help="Sets the number of Optuna Trials to do")
    parser.add_argument('--optuna_use_best_trial', default=False, action='store_true', help="Use the best trial from the database passed as optuna_storage")
    parser.add_argument('--optuna_storage', default=None, type=str, help="An Optuna DB file")
    parser.add_argument('--optuna_study_name', default=None, type=str, help="Name of the optuna study in the Optuna DB file")

    # Input data normalization
    parser.add_argument('--data_mean', default=None, type=float, required=False, help="Mean of the input data. Used for input normalization. E.g. in the case of CT scans, this would be the mean CT value over all scans. Note: normalization is only performed if both data_mean and data_stddev are defined.")
    parser.add_argument('--data_stddev', default=None, type=float, required=False, help="Standard deviation of the input data. Used for input normalization. E.g. in the case of CT scans, this would be the standard deviation of CT values over all scans. Note: normalization is only performed if both data_mean and data_stddev are defined.")


    args = parser.parse_args()

    if args.horovod or args.optuna_distributed:
        hvd.init()
        np.random.seed(args.seed + hvd.rank())
        tf.random.set_random_seed(args.seed + hvd.rank())
        random.seed(args.seed + hvd.rank())

        print(f"Rank {hvd.rank()}:{hvd.local_rank()} reporting!")

    else:
        np.random.seed(args.seed)
        tf.random.set_random_seed(args.seed)
        random.seed(args.seed)

    if args.horovod:
        verbose = hvd.rank() == 0
    else:
        verbose = True



    # if args.coninue_path:
    #     assert args.load_phase is not None, "Please specify in which phase the weights of the " \
    #                                         "specified continue_path should be loaded."

    # Set default for *_rise_niter and *_decay_niter if needed. We can't do this natively with ArgumentParser because it depends on the value of another argument.
    if args.g_lr_increase and not args.g_lr_rise_niter:
        args.g_lr_rise_niter = int(args.mixing_nimg/2)
        if verbose:
            print(f"Increasing learning rate requested for the generator, but no number of iterations was specified for the increase (g_lr_rise_niter). Defaulting to {args.g_lr_rise_niter}.")
    if args.g_lr_decrease and not args.g_lr_decay_niter:
        args.g_lr_decay_niter = int(args.stabilizing_nimg/2)
        if verbose:
            print(f"Decreasing learning rate requested for the generator, but no number of iterations was specified for the increase (g_lr_decay_niter). Defaulting to {args.g_lr_decay_niter}.")
    if args.d_lr_increase and not args.d_lr_rise_niter:
        args.d_lr_rise_niter = int(args.mixing_nimg/2)
        if verbose:
            print(f"Increasing learning rate requested for the discriminator, but no number of iterations was specified for the increase (d_lr_rise_niter). Defaulting to {args.d_lr_rise_niter}.")
    if args.d_lr_decrease and not args.d_lr_decay_niter:
        args.d_lr_decay_niter = int(args.stabilizing_nimg/2)
        if verbose:
            print(f"Decreasing learning rate requested for the discriminator, but no number of iterations was specified for the increase (d_lr_decay_niter). Defaulting to {args.d_lr_decay_niter}.")

    if args.architecture in ('stylegan2'):
        assert args.starting_phase == args.ending_phase

    if 'OMP_NUM_THREADS' not in os.environ:
        print("Warning: OMP_NUM_THREADS not set. Setting it to 1.")
        os.environ['OMP_NUM_THREADS'] = str(1)

    gopts = tf.GraphOptions(place_pruned_graph=True)
    config = tf.ConfigProto(graph_options=gopts, allow_soft_placement=True)
    # config = tf.ConfigProto()

    if args.gpu:
        config.gpu_options.allow_growth = True
        # config.inter_op_parallelism_threads = 1
        #config.gpu_options.per_process_gpu_memory_fraction = 0.96
        if args.horovod or args.optuna_distributed:
            config.gpu_options.visible_device_list = str(hvd.local_rank())

    else:
        config = tf.ConfigProto(graph_options=gopts,
                                intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']),
                                inter_op_parallelism_threads=args.num_inter_ops,
                                allow_soft_placement=True,
                                device_count={'CPU': int(os.environ['OMP_NUM_THREADS'])})


    main(args, config)
