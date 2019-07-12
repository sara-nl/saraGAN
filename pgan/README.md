# Progressive GAN (Karras 2018)

## Description
This repo contains the current code for our implementation of Karras 2018.
Since I think the readability of pure Tensorflow code is terrible, I tried to build it using Keras as much as possible. 
Where Tensorflow was needed, I build it inside a Keras wrapper.

Repo's used as inspiration.
https://github.com/johnryh/Face_Embedding_GAN
https://github.com/tkarras/progressive_growing_of_gans
https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py

## Files
*/architectures/*: Directory with files for different architectures.

*loss.py*: Wasserstein GP loss from Gulrajani and the DriftLoss from Karras.

*initializers.py*:  Weight initializer from the paper and from the repo. 'Gain' values throughout the code are taken from the NVIDIA implementation.

*layers.py*: PixelNorm, MinibatchSTDDev, DiscrminatorBlock, GeneratorBlock, InterpolationLayer, AlphaMixingLayer

*main.py*: Main model construction and training. 



## To Do
* Change 'channels' in the arthitecture and num_channels in the Discriminator/Generator Blocks to 'filters' for consistency with Keras.
* Separate cluttered main.py file into different files.
* Debug training.
* Get training on CIFAR10 working.
* Implement progressive growing. Note: the network architecture construction already works by just increasing phase. What needs to be done is re-loading the weights from the previous phase in the correct layers and implementing the alpha-mixing layer.
* Get training on CelebA and x-rays.
* Check if horovod implementation is working correctly.
* Implement conditioning.
* Check conditional training results.
* Go to 3D.
