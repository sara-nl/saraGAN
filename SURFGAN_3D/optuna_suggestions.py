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
    if not args.g_lr:
        args.g_lr = trial.suggest_loguniform('generator_LR', 1e-6, 1e-2)
    if not args.d_lr:
        args.d_lr = trial.suggest_loguniform('discriminator_LR', 1e-6, 1e-2)

    lr_schedule = get_predefined_lr_schedules()
    if not args.g_lr_increase and not args.g_lr_rise_niter:
        g_lr_sched_inc = trial.suggest_categorical('g_lr_sched_inc', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        args.g_lr_increase = lr_schedule[g_lr_sched_inc]['lr_sched']
        args.g_lr_rise_niter = np.ceil(lr_schedule[g_lr_sched_inc]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
    elif (args.g_lr_increase and not args.g_lr_rise_niter) or (args.g_lr_rise_niter and not args.g_lr_increase):
        if verbose:
            print("ERROR: either both g_lr_increase and g_lr_rise_niter have to be specified, or neither. You cannot specify only one.")
            raise NotImplementedError()
    
    if not args.g_lr_decrease and not args.g_lr_decay_niter:
        g_lr_sched_dec = trial.suggest_categorical('g_lr_sched_dec', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        args.g_lr_decrease = lr_schedule[g_lr_sched_dec]['lr_sched']
        args.g_lr_decay_niter = np.ceil(lr_schedule[g_lr_sched_dec]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
    elif (args.g_lr_decrease and not args.g_lr_decay_niter) or (args.g_lr_decay_niter and not args.g_lr_decrease):
        if verbose:
            print("ERROR: either both g_lr_decrease and g_lr_decay_niter have to be specified, or neither. You cannot specify only one.")
            raise NotImplementedError()

    if not args.d_lr_increase and not args.d_lr_rise_niter:
        d_lr_sched_inc = trial.suggest_categorical('d_lr_sched_inc', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        args.d_lr_increase = lr_schedule[d_lr_sched_inc]['lr_sched']
        args.d_lr_rise_niter = np.ceil(lr_schedule[d_lr_sched_inc]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
    elif (args.d_lr_increase and not args.d_lr_rise_niter) or (args.d_lr_rise_niter and not args.d_lr_increase):
        if verbose:
            print("ERROR: either both d_lr_increase and d_lr_rise_niter have to be specified, or neither. You cannot specify only one.")
            raise NotImplementedError()
    
    if not args.d_lr_decrease and not args.d_lr_decay_niter:
        d_lr_sched_dec = trial.suggest_categorical('d_lr_sched_dec', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        args.d_lr_decrease = lr_schedule[d_lr_sched_dec]['lr_sched']
        args.d_lr_decay_niter = np.ceil(lr_schedule[d_lr_sched_dec]['lr_fract'] * (args.mixing_nimg + args.stabilizing_nimg)).astype(np.int32)
    elif (args.d_lr_decrease and not args.d_lr_decay_niter) or (args.d_lr_decay_niter and not args.d_lr_decrease):
        if verbose:
            print("ERROR: either both d_lr_decrease and d_lr_decay_niter have to be specified, or neither. You cannot specify only one.")
            raise NotImplementedError()
        
    return args