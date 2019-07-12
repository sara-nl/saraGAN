# Keys: train phases.

# How much to increase the kernel size in the final layer of the discriminator to get a [n_channels,1,1] vector.
CELEBA_KERNEL_OFFSET = 1

generator_architecture = {
    # Input block: Latent -> Dense -> Conv -> Conv
    1: {'channels': [512, 512],
        'filters': [3, 3]},
    2: {'channels': [512, 512],
        'filters': [3, 3]},
    3: {'channels': [512, 512],
        'filters': [3, 3]},
    4: {'channels': [512, 512],
        'filters': [3, 3]},
    5: {'channels': [256, 256],
        'filters': [3, 3]},
    6: {'channels': [128, 128],
        'filters': [3, 3]},
    7: {'channels': [64, 64],
        'filters': [3, 3]},
    8: {'channels': [32, 32],
        'filters': [3, 3]},
    9: {'channels': [16, 16],
        'filters': [3, 3]}
}

discriminator_architecture = {
    # Input block: Input -> Conv(1x1) -> Conv1 -> Conv2 -> Downsample
    1: {'channels': [16, 32],
        'filters': [3, 3]},
    # Discrimintaor Block: Downsample -> Conv -> Conv
    2: {'channels': [32, 64],
        'filters': [3, 3]},
    3: {'channels': [64, 128],
        'filters': [3, 3]},    
    4: {'channels': [128, 256],
        'filters': [3, 3]},    
    5: {'channels': [256, 512],
        'filters': [3, 3]},    
    6: {'channels': [512, 512],
        'filters': [3, 3]},    
    7: {'channels': [512, 512],
        'filters': [3, 3]},    
    8: {'channels': [512, 512],
        'filters': [3, 3]},    
    9: {'channels': [512, 512],
        'filters': [3, 3]}
}
