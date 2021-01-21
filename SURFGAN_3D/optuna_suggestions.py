import numpy as np
from utils import get_verbosity

def get_predefined_lr_schedules():
    lr_schedules = [
            {'lr_sched': None, 'lr_fract': 0.5},
            {'lr_sched': 'linear', 'lr_fract': 0.125},
            {'lr_sched': 'linear', 'lr_fract': 0.25},
            {'lr_sched': 'linear', 'lr_fract': 0.375},
            {'lr_sched': 'linear', 'lr_fract': 0.5},
            {'lr_sched': 'exponential', 'lr_fract': 0.125},
            {'lr_sched': 'exponential', 'lr_fract': 0.25},
            {'lr_sched': 'exponential', 'lr_fract': 0.375},
            {'lr_sched': 'exponential', 'lr_fract': 0.5},
        ]
    return lr_schedules

def optuna_override_undefined(args, trial):
    """Let optuna suggest values for variables for which no argument was specified (args.*=None).
    Parameters:
    -----------
      args: the original arguments passed to main.
      trial: the current optuna trial. Different values are sampled for each trial.

    Returns:
    -----------
      args: the argument list, where values of None have been filled by optuna suggested values
    """
    verbose = get_verbosity(args.horovod, args.optuna_distributed)

    if not args.base_batch_size:
        args.base_batch_size = 2 ** trial.suggest_int('base_batch_size_exponent', 1, 6)
        if verbose:
            print(f"args.base_batch_size = {args.base_batch_size} (from: optuna trial)")
    elif verbose:
        print(f"args.base_batch_size = {args.base_batch_size} (from: command line argument)")

    if not args.g_lr:
        args.g_lr = trial.suggest_loguniform('generator_LR', 1e-6, 1e-2)
        if verbose:
            print(f"args.g_lr = {args.g_lr} (from: optuna trial)")
    elif verbose:
        print(f"args.g_lr = {args.g_lr} (from: command line argument)")

    if not args.d_lr:
        args.d_lr = trial.suggest_loguniform('discriminator_LR', 1e-6, 1e-2)
        if verbose:
            print(f"args.d_lr = {args.d_lr} (from: optuna trial)")
    elif verbose:
        print(f"args.d_lr = {args.d_lr} (from: command line argument)")

    lr_schedule = get_predefined_lr_schedules()
    if args.g_lr_increase is None and args.g_lr_rise_niter is None:
        g_lr_sched_inc = trial.suggest_categorical('g_lr_sched_inc', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        args.g_lr_increase = lr_schedule[g_lr_sched_inc]['lr_sched']
        args.g_lr_rise_niter = np.ceil(lr_schedule[g_lr_sched_inc]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
        if verbose:
            print(f"args.g_lr_increase = {args.g_lr_increase} (from: optuna trial)")
            print(f"args.g_lr_rise_niter = {args.g_lr_rise_niter} (from: optuna trial)")
    elif (args.g_lr_increase is not None and args.g_lr_rise_niter is None):
        if verbose:
            print("ERROR: if you specify g_lr_increase on the command line, g_lr_rise_niter also has to be specified.")
        raise NotImplementedError()
    elif verbose:
        print(f"args.g_lr_increase = {args.g_lr_increase} (from: command line argument)")
        print(f"args.g_lr_rise_niter = {args.g_lr_rise_niter} (from: command line argument)")
    
    if  args.g_lr_decrease is None and args.g_lr_decay_niter is None:
        g_lr_sched_dec = trial.suggest_categorical('g_lr_sched_dec', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        args.g_lr_decrease = lr_schedule[g_lr_sched_dec]['lr_sched']
        args.g_lr_decay_niter = np.ceil(lr_schedule[g_lr_sched_dec]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
        if verbose:
            print(f"args.g_lr_decrease = {args.g_lr_decrease} (from: optuna trial)")
            print(f"args.g_lr_decay_niter = {args.g_lr_decay_niter} (from: optuna trial)")
    elif (args.g_lr_decrease is not None and args.g_lr_decay_niter is None):
        if verbose:
            print("ERROR: if you specify g_lr_decrease on the command line, g_lr_decay_niter also has to be specified.")
        raise NotImplementedError()
    elif verbose:
        print(f"args.g_lr_decrease = {args.g_lr_decrease} (from: command line argument)")
        print(f"args.g_lr_decay_niter = {args.g_lr_decay_niter} (from: command line argument)")

    if args.d_lr_increase is None and args.d_lr_rise_niter is None:
        d_lr_sched_inc = trial.suggest_categorical('d_lr_sched_inc', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        args.d_lr_increase = lr_schedule[d_lr_sched_inc]['lr_sched']
        args.d_lr_rise_niter = np.ceil(lr_schedule[d_lr_sched_inc]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
        if verbose:
            print(f"args.d_lr_increase = {args.d_lr_increase} (from: optuna trial)")
            print(f"args.d_lr_rise_niter = {args.d_lr_rise_niter} (from: optuna trial)")
    elif (args.d_lr_increase is not None and args.d_lr_rise_niter is None):
        if verbose:
            print("ERROR: if you specify d_lr_increase on the command line, d_lr_rise_niter also has to be specified.")
        raise NotImplementedError()
    elif verbose:
        print(f"args.d_lr_increase = {args.d_lr_increase} (from: command line argument)")
        print(f"args.d_lr_rise_niter = {args.d_lr_rise_niter} (from: command line argument)")
    
    if args.d_lr_decrease is None and args.d_lr_decay_niter is None:
        d_lr_sched_dec = trial.suggest_categorical('d_lr_sched_dec', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        args.d_lr_decrease = lr_schedule[d_lr_sched_dec]['lr_sched']
        args.d_lr_decay_niter = np.ceil(lr_schedule[d_lr_sched_dec]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
        if verbose:
            print(f"args.d_lr_decrease = {args.d_lr_decrease} (from: optuna trial)")
            print(f"args.d_lr_decay_niter = {args.d_lr_decay_niter} (from: optuna trial)")
    elif (args.d_lr_decrease is not None and args.d_lr_decay_niter is None):
        if verbose:
            print("ERROR: if you specify d_lr_decrease on the command line, d_lr_decay_niter also has to be specified.")
        raise NotImplementedError()
    elif verbose:
        print(f"args.d_lr_decrease = {args.d_lr_decrease} (from: command line argument)")
        print(f"args.d_lr_decay_niter = {args.d_lr_decay_niter} (from: command line argument)")
        
    return args