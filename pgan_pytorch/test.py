import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import os
import os.path
import sys
import matplotlib.pyplot as plt
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.init import calculate_gain, _calculate_correct_fan, _calculate_fan_in_and_fan_out



def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir, extensions=None, is_valid_file=None):
    paths = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
        
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                paths.append(path)
                    
    return paths


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/xxx.ext
        root/xxy.ext
        root/xxz.ext
        root/123.ext
        root/nsdf3.ext
        root/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        samples = make_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders in: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)
    
    
dataset = DatasetFolder('/project/davidr/lidc_idri/npys/lanczos/8x8/', 
                               loader=lambda path: np.load(path),
                               extensions=('npy',),
                               transform=lambda x: torch.from_numpy(x))


def num_filters(phase, num_phases, base_dim):
    num_downscales = int(np.log2(base_dim / 16))
    filters = min(base_dim // (2 ** (phase - num_phases + num_downscales)), base_dim)
    return filters


class ScaleLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, gain):
        super(ScaleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
        self.std = gain / np.sqrt(fan_in)

    def forward(self, input):
        return F.linear(input, self.weight * self.std, self.bias)
    

class ScaleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gain, stride=1,
                 padding=0):
        super(ScaleConv3d, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(
                out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
                
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
        self.std = gain / np.sqrt(fan_in)
        

    def forward(self, input):        
        return F.conv3d(input, self.weight * self.std, self.bias, self.stride,
                        self.padding)

    
class DiscriminatorBlock(nn.Sequential):
    def __init__(self, filters_in, filters_out):
        super(DiscriminatorBlock, self).__init__()
        
        self.conv1 = ScaleConv3d(filters_in, filters_in, 3, padding=1, gain=np.sqrt(2))
        self.conv2 = ScaleConv3d(filters_in, filters_out, 3, padding=1, gain=np.sqrt(2))
        self.lrelu = nn.LeakyReLU()
        self.downsampling = nn.AvgPool3d(2)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.downsampling(x)
        return x
        
        
class FromRGB(nn.Sequential):
    def __init__(self, filters):
        super(FromRGB, self).__init__()
        self.fromrgb = nn.Sequential(
            ScaleConv3d(1, filters, 3, padding=1, gain=np.sqrt(2)),
            nn.LeakyReLU()
        )
    
    def forward(self, input):
        return self.fromrgb(input)

    
class MinibatchStandardDeviation(nn.Module):
    def __init__(self, group_size=4):
        super(MinibatchStandardDeviation, self).__init__()
        self.group_size = group_size
        
    def forward(self, input):
        group_size = min(self.group_size, input.shape[0])
        if group_size < len(input):
            for i in range(group_size, len(input) + 1):
                if len(input) % i == 0:
                    group_size = i
                    break
        
        s = input.shape
        y = input.view([group_size, -1, s[1], s[2], s[3], s[4]])
        y -= torch.mean(y, dim=0, keepdim=True)                     
        y = torch.mean(y ** 2, dim=0)                           
        y = torch.sqrt(y + 1e-8)
        y = torch.mean(y, dim=[1, 2, 3, 4], keepdim=True)
        y = y.repeat([group_size, 1, s[2], s[3], s[4]])
        return torch.cat([input, y], dim=1)                            

    
class Discriminator(nn.Module):
    def __init__(self, phase, num_phases, base_dim, latent_dim, base_shape):
        super(Discriminator, self).__init__()
        filters_out = num_filters(phase, num_phases, base_dim)
        self.phase = phase
        self.fromrgb = FromRGB(filters_out)
        self.blocks = nn.ModuleDict()
        self.fromrgbs = nn.ModuleDict()
        for i in reversed(range(1, phase)):
            filters_in = num_filters(i + 1, num_phases, base_dim)
            filters_out = num_filters(i, num_phases, base_dim)
            self.blocks[f'discriminator_block_{i + 1}'] = DiscriminatorBlock(filters_in, filters_out)
            self.fromrgbs[f'from_rgb_{i}'] = FromRGB(filters_out)
            
        self.downscale = nn.AvgPool3d(2)
            
        self.discriminator_out = nn.Sequential(
            MinibatchStandardDeviation(),
            ScaleConv3d(filters_out + 1, base_dim, 3, padding=1, gain=np.sqrt(2)),
            nn.LeakyReLU(),
            nn.Flatten(),
            ScaleLinear(np.product(base_shape) * base_dim, latent_dim, gain=np.sqrt(2)),
            nn.LeakyReLU(),
            ScaleLinear(latent_dim, 1, gain=1)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, input, alpha=0):
        
        input = input.to(self.device)
        
        x_downscale = input.clone()
        
        x = self.fromrgb(input)
                                
        for i in reversed(range(1, self.phase)):
            x = self.blocks[f'discriminator_block_{i + 1}'](x)
            
            x_downscale = self.downscale(x_downscale)
            fromrgb_prev = self.fromrgbs[f'from_rgb_{i}'](x_downscale)
            x = alpha * fromrgb_prev + (1 - alpha) * x
            
        x = self.discriminator_out(x)
        return x
            
        
class ChannelNormalization(nn.Module):
    def __init__(self):
        super(ChannelNormalization, self).__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class GeneratorBlock(nn.Sequential):
    def __init__(self, filters_in, filters_out):
        super(GeneratorBlock, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv1 = ScaleConv3d(filters_in, filters_out, 3, padding=1, gain=np.sqrt(2))
        self.conv2 = ScaleConv3d(filters_out, filters_out, 3, padding=1, gain=np.sqrt(2))
        self.lrelu = nn.LeakyReLU()
        self.cn = ChannelNormalization()
    
    def forward(self, input):
        x = self.upsampling(input)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.cn(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.cn(x)
        return x
        
        
class ToRGB(nn.Sequential):
    def __init__(self, filters_in, channels=1):
        super(ToRGB, self).__init__()
        self.conv = ScaleConv3d(filters_in, channels, 3, padding=1, gain=1)
        
    def forward(self, input):
        return self.conv(input)
    
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return torch.reshape(input, self.shape)
    
class Generator(nn.Module):
    def __init__(self, phase, num_phases, base_dim, latent_dim, base_shape):
        super(Generator, self).__init__()
        self.phase = phase
        self.latent_dim = latent_dim
        filters = base_dim
        self.generator_in = nn.Sequential(
            ScaleLinear(latent_dim, np.product(base_shape) * filters, gain=np.sqrt(2) / 4),
            nn.LeakyReLU(),
            Reshape([-1, filters] + list(base_shape)),
            ScaleConv3d(filters, filters, 3, padding=1, gain=np.sqrt(2)),
            nn.LeakyReLU(),
            ChannelNormalization()
        )
        
        self.to_rgb_1 = ToRGB(filters)
        
        self.blocks = nn.ModuleDict()
        self.to_rgbs = nn.ModuleDict()
        
        for i in range(1, phase):
            filters_in = num_filters(i, num_phases, base_dim)
            filters_out = num_filters(i + 1, num_phases, base_dim)
            self.blocks[f'generator_block_{i + 1}'] = GeneratorBlock(filters_in, filters_out)
            self.to_rgbs[f'to_rgb_{i + 1}'] = ToRGB(filters_out)
            
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
            
    
    def forward(self, input, alpha=0):
        input = input.to(self.device)
        
        x = self.generator_in(input)
        
        images_out = self.to_rgb_1(x)
        
        for i in range(1, self.phase):
            x = self.blocks[f'generator_block_{i + 1}'](x)
            img_gen = self.to_rgbs[f'to_rgb_{i + 1}'](x)
            images_out = alpha * (self.upsample(images_out)) + (1 - alpha) * img_gen
        
        return images_out
    
    
discriminator = Discriminator(2, 8, 256, 256, (1, 4, 4))
generator = Generator(2, 8, 256, 256, (1, 4, 4))

d_optim = torch.optim.Adam(discriminator.parameters(), betas=(0, 0.9))
g_optim = torch.optim.Adam(generator.parameters(), betas=(0, 0.9))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, 
                                          shuffle=True,
                                          num_workers=4,
                                          pin_memory=True)

def wasserstein_loss(y_pred):
    return y_pred.mean()

def gradient_penalty_loss(y_pred, 
                          averaged_samples,
                          gradient_penalty_weight=10):
    
    grad_outputs = torch.ones_like(y_pred).to(y_pred.device)
    gradients = torch.autograd.grad(
        y_pred, 
        inputs=averaged_samples,
        grad_outputs=grad_outputs,
        create_graph=True, 
        retain_graph=True)[0]
    
    gradients_sqr = gradients ** 2
    gradients_sqr_sum = torch.sum(gradients_sqr, 
                                  dim=tuple(range(1, len(gradients_sqr.shape))))
    gradient_l2_norm = torch.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * (1 - gradient_l2_norm) ** 2
    return torch.mean(gradient_penalty)

def random_weighted_average(tensor_a, tensor_b):
    weights = torch.rand_like(tensor_a).to(tensor_a.device)
    return weights * tensor_a + (1 - weights) * tensor_b


def train_epoch(data_loader, generator, discriminator, generator_optim, discriminator_optim, alpha):
    
    d_losses = []
    g_losses = []
    
    for i, x_real in enumerate(data_loader):
        
        # Train discriminator.
        generator.eval()
        discriminator.train()
        
        x_real = x_real.to(discriminator.device)
        z = torch.randn(x_real.shape[0], generator.latent_dim)
        x_fake = generator(z, alpha)
        
        d_real = discriminator(x_real, alpha)
        d_fake = discriminator(x_fake, alpha)
        
        averaged_samples = torch.autograd.Variable(random_weighted_average(x_fake, x_real),
                                          requires_grad=True)
        d_avg = discriminator(averaged_samples, alpha)
        
        gp_loss = gradient_penalty_loss(d_avg, averaged_samples)
        real_loss = wasserstein_loss(d_real)
        fake_loss = wasserstein_loss(d_fake)
        
        d_loss = real_loss - fake_loss + gp_loss
        
        discriminator_optim.zero_grad()
        d_loss.backward()
        discriminator_optim.step()
        
        d_losses.append(d_loss.item())
        
        # Train generator.
        generator.train()
        discriminator.eval()
        
        z = torch.randn(x_real.shape[0], generator.latent_dim)
        x_fake = generator(z)
        d_fake = discriminator(x_fake)
        g_loss = wasserstein_loss(d_fake)
        
        generator_optim.zero_grad()
        g_loss.backward()
        generator_optim.step()
        
        g_losses.append(g_loss.item())
        
        if i % 50 == 0:
            print(g_loss.item(), d_loss.item())
           
    print("Real")
    print(x_real.shape)
    plt.imshow(x_real[0, 0, 0].squeeze().cpu().detach().numpy())
    plt.savefig('real.png')

    print("Fake")
    plt.imshow(x_fake[0, 0, 0].squeeze().cpu().detach().numpy())
    plt.savefig('fake.png')
    
    return np.mean(d_losses), np.mean(g_losses)
        
    
for i in range(256):
    print(i)
    train_epoch(data_loader, generator, discriminator, generator_optim=g_optim, 
            discriminator_optim=d_optim, alpha=1)