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

    # base_batch_size
    if not args.base_batch_size:
        args.base_batch_size = 2 ** trial.suggest_int('base_batch_size_exponent', 1, 6)
        if verbose:
            print(f"args.base_batch_size = {args.base_batch_size} (from: optuna trial)")
    elif verbose:
        print(f"args.base_batch_size = {args.base_batch_size} (from: command line argument)")

    # g_lr
    if not args.g_lr:
        args.g_lr = trial.suggest_loguniform('generator_LR', 1e-2, 1e-1)
        if verbose:
            print(f"args.g_lr = {args.g_lr} (from: optuna trial)")
    elif verbose:
        print(f"args.g_lr = {args.g_lr} (from: command line argument)")

    # d_lr
    if not args.d_lr:
        args.d_lr = trial.suggest_loguniform('discriminator_LR', 1e-3, 5e-2)
        if verbose:
            print(f"args.d_lr = {args.d_lr} (from: optuna trial)")
    elif verbose:
        print(f"args.d_lr = {args.d_lr} (from: command line argument)")

    # g_lr_increase, g_lr_rise_niter
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
    
    # g_lr_decrease, g_lr_decay_niter
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

    # d_lr_increase, d_lr_rise_niter
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
    
    # d_lr_decreaes, d_lr_decay_niter
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

    # optimizer, d_optimizer (if not args.d_use_different_optimizer)
    d_optimizer_set = False
    if args.optimizer is None:
        args.optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'Adadelta'])
        if verbose:
            print(f"args.optimizer = {args.optimizer} (from: optuna trial)")
        # Use same sampled optimizer for discriminator
        if not args.d_use_different_optimizer:
            args.d_optimizer = args.optimizer
            d_optimizer_set = True
            if verbose:
                print(f"args.d_optimizer = {args.d_optimizer} (same as args.optimizer)")
    elif verbose:
        print(f"args.optimizer = {args.optimizer} (from: command line argument)")

    # d_optimizer (if args.d_use_different_optimizer)
    if args.d_optimizer is None and args.d_use_different_optimizer:
        args.d_optimizer = trial.suggest_categorical('d_optimizer', ['Adam', 'SGD', 'Momentum', 'Adadelta'])
        if verbose:
            print(f"args.d_optimizer = {args.d_optimizer} (from: optuna trial)")       
    elif verbose and not d_optimizer_set: # Check that args.d_optimizer wasn't set yet to the same value as args.optimizer - if so, it shouldn't print here
        print(f"args.d_optimizer = {args.d_optimizer} (from: command line argument)")

    # adam_beta1, d_adam_beta1 (if not args.d_use_different_beta1)
    d_adam_beta1_set = False
    if args.adam_beta1 is None:
        args.adam_beta1 = trial.suggest_float("adam_beta1", 0, 0.4)
        if verbose:
            print(f"args.adam_beta1 = {args.adam_beta1} (from: optuna trial)")
        # Use sampled adam_beta for discriminator
        if not args.d_use_different_beta1:
            args.d_adam_beta1 = args.adam_beta1
            d_adam_beta1_set = True
            if verbose:
                print(f"args.d_adam_beta1 = {args.d_adam_beta1} (same as args.adam_beta1)")
    elif verbose:
        print(f"args.adam_beta1 = {args.adam_beta1} (from: command line argument)")

    # d_adam_beta1 (if args.d_use_different_beta1)
    if args.d_adam_beta1 is None and args.d_use_different_beta1:
        args.d_adam_beta1 = trial.suggest_float("d_adam_beta1", 0, 0.4)
        if verbose:
            print(f"args.d_adam_beta1 = {args.d_adam_beta1} (from: optuna trial)")
    elif verbose and not d_adam_beta1_set: # Check that args.d_adam_beta1 wasn't set yet to the same value as args.adam_beta1 - if so, it shouldn't print here
        print(f"args.d_adam_beta1 = {args.d_adam_beta1} (from: command line argument)")

    # adam_beta2, d_adam_beta2 (if not args.d_use_different_beta2)
    d_adam_beta2_set = False
    if args.adam_beta2 is None:
        args.adam_beta2 = trial.suggest_float("adam_beta2", 0.75, 1)
        if verbose:
            print(f"args.adam_beta2 = {args.adam_beta2} (from: optuna trial)")
        # Use sampled adam_beta for discriminator
        if not args.d_use_different_beta2:
            args.d_adam_beta2 = args.adam_beta2
            d_adam_beta2_set = True
            if verbose:
                print(f"args.d_adam_beta2 = {args.d_adam_beta2} (same as args.adam_beta2)")
    elif verbose:
        print(f"args.adam_beta2 = {args.adam_beta2} (from: command line argument)")

    # d_adam_beta2 (if args.d_use_different_beta2)
    if args.d_adam_beta2 is None and args.d_use_different_beta2:
        args.d_adam_beta2 = trial.suggest_float("d_adam_beta2", 0.75, 1)
        if verbose:
            print(f"args.d_adam_beta2 = {args.d_adam_beta2} (from: optuna trial)")
    elif verbose and not d_adam_beta2_set: # Check that args.d_adam_beta2 wasn't set yet to the same value as args.adam_beta2 - if so, it shouldn't print here
        print(f"args.d_adam_beta2 = {args.d_adam_beta2} (from: command line argument)")

    # rho, d_rho (if not args.d_use_different_rho)
    d_rho_set = False
    if args.rho is None:
        args.rho = trial.suggest_float("adadelta_rho", 0, 1)
        if verbose:
            print(f"args.rho = {args.rho} (from: optuna trial)")
        # Use sampled rho for discriminator
        if not args.d_use_different_rho:
            args.d_rho = args.rho
            d_rho_set = True
            if verbose:
                print(f"args.d_rho = {args.d_rho} (same as args.rho)")
    elif verbose:
        print(f"args.rho = {args.rho} (from: command line argument)")

    # d_rho (if args.d_use_different_rho)
    if args.d_rho is None and args.d_use_different_rho:
        args.d_rho = trial.suggest_float("d_adadelta_rho", 0, 1)
        if verbose:
            print(f"args.d_rho = {args.d_rho} (from: optuna trial)")
    elif verbose and not d_rho_set: # Check that args.d_rho wasn't set yet to the same value as args.rho - if so, it shouldn't print here
        print(f"args.d_rho = {args.d_rho} (from: command line argument)")

    # momentum, d_momentum (if not args.d_use_different_momentum)
    d_momentum_set = False
    if args.momentum is None:
        args.momentum = trial.suggest_float("SGD_momentum", 0, 1)
        if verbose:
            print(f"args.momentum = {args.momentum} (from: optuna trial)")
        # Use sampled momentum for discriminator
        if not args.d_use_different_momentum:
            args.d_momentum = args.momentum
            d_momentum_set = True
            if verbose:
                print(f"args.d_momentum = {args.d_momentum} (same as args.momentum)")
    elif verbose:
        print(f"args.momentum = {args.momentum} (from: command line argument)")

    # d_momentum (if args.d_use_different_momentum)
    if args.d_momentum is None and args.d_use_different_momentum:
        args.d_momentum = trial.suggest_float("d_SGD_momentum", 0, 1)
        if verbose:
            print(f"args.d_momentum = {args.d_momentum} (from: optuna trial)")
    elif verbose and not d_momentum_set:
        print(f"args.d_momentum = {args.d_momentum} (from: command line argument)")

    for i in range(0, len(args.conv_kernel_size)):
        if args.conv_kernel_size[i] is None:
            # Suggest some odd number as kernel size.
            args.conv_kernel_size[i] = trial.suggest_int(f"Kernel_size_{i}", 1, 9, 2)
            if verbose:
                print(f"args.conv_kernel_size[{i}] = {args.conv_kernel_size[i]} (from: optuna trial)")
        else:
            if verbose:
                print(f"args.conv_kernel_size[{i}] = {args.conv_kernel_size[i]} (from: command line argument)")

    # filter_spec
    max_filter_counts = [9, 8, 7, 7, 6, 5, 4] # corresponds to filter counts of [512, 256, 128, 128, 64, 32, 16]
    for phase_i in range(0, len(args.filter_spec)):
        for conv_j in range(0, len(args.filter_spec[phase_i])):
                if args.filter_spec[phase_i][conv_j] is None or args.filter_spec[phase_i][conv_j] == "None":
                    # Suggest some power of two as number of filters
                    args.filter_spec[phase_i][conv_j] = 2 ** trial.suggest_int(f"Filter_count_exponent_{phase_i}_{conv_j}", 2, max_filter_counts[phase_i])
                    if verbose:
                        print(f"args.filter_spec[{phase_i}][{conv_j}] = {args.filter_spec[phase_i][conv_j]} (from: optuna trial)")
                else:
                    if verbose:
                        print(f"args.filter_spec[{phase_i}][{conv_j}] = {args.filter_spec[phase_i][conv_j]} (from: command line argument)")

    # kernel_spec
    for phase_i in range(0, len(args.kernel_spec)):
        for conv_j in range(0, len(args.kernel_spec[phase_i])):
            for kernel_k in range(0, len(args.kernel_spec[phase_i][conv_j])):
                if args.kernel_spec[phase_i][conv_j][kernel_k] is None or args.kernel_spec[phase_i][conv_j][kernel_k] == "None":
                    # Suggest some odd number as kernel size
                    if args.optuna_square_kernels: # pick kernel size the same along all dimensions
                        if kernel_k == 0:
                            args.kernel_spec[phase_i][conv_j][kernel_k] = trial.suggest_int(f"Kernel_size_{phase_i}_{conv_j}_{kernel_k}", 1, 7, 2)
                        else:
                            args.kernel_spec[phase_i][conv_j][kernel_k] = args.kernel_spec[phase_i][conv_j][0]
                    else:
                        args.kernel_spec[phase_i][conv_j][kernel_k] = trial.suggest_int(f"Kernel_size_{phase_i}_{conv_j}_{kernel_k}", 1, 7, 2)
                    if verbose:
                        print(f"args.kernel_spec[{phase_i}][{conv_j}][{kernel_k}] = {args.kernel_spec[phase_i][conv_j][kernel_k]} (from: optuna trial)")
                else:
                    if verbose:
                        print(f"args.kernel_spec[{phase_i}][{conv_j}][{kernel_k}] = {args.kernel_spec[phase_i][conv_j][kernel_k]} (from: command line argument)")
        
    return args