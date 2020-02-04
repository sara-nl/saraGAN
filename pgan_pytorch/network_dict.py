import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from torch.nn.init import calculate_gain, _calculate_correct_fan
from torch.nn.modules.utils import _triple


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


LEAKINESS = 0.3
NONLINEARITY_DICT = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(LEAKINESS),
    'swish': Swish()
}

def num_filters(phase, num_phases, base_dim):
    num_downscales = int(np.log2(base_dim / 16))
    filters = min(base_dim // (2 ** (phase - num_phases + num_downscales)), base_dim)
    return filters


def kaiming_normal_(tensor, gain_mode, mode='fan_in', param=None):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(gain_mode, param)
    std = gain / np.sqrt(fan)
    with torch.no_grad():
        tensor.normal_(0, 1)
        return std
    
    
class EqualizedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinearity, stride=1,
                 padding=0, param=None):
        super(EqualizedConv3d, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.nonlinearity = nonlinearity
        if nonlinearity == 'leaky_relu':
            assert param is not None
        self.param = param

        self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.std = None  # Placeholder
        self.reset_parameters()

    def reset_parameters(self):
        self.std = kaiming_normal_(self.weight, self.nonlinearity, param=self.param)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.conv3d(input, self.weight * self.std, self.bias, self.stride,
                        self.padding)

    
class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity, param=None):
        super(EqualizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.std = None  # Placeholder
        self.nonlinearity = nonlinearity
        self.param = param
        if nonlinearity == 'leaky_relu':
            assert param is not None

        self.reset_parameters()

    def reset_parameters(self):
        self.std = kaiming_normal_(self.weight, gain_mode=self.nonlinearity, param=self.param)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight * self.std, self.bias)

    
class DiscriminatorBlock(nn.Sequential):
    def __init__(self, filters_in, filters_out, nonlinearity, param=None):
        super(DiscriminatorBlock, self).__init__()
        
        self.conv1 = EqualizedConv3d(filters_in, filters_in, 3, padding=1,
                                     nonlinearity=nonlinearity, param=param)
        self.conv2 = EqualizedConv3d(filters_in, filters_out, 3, padding=1,
                                     nonlinearity=nonlinearity, param=param)
        self.act = activation(nonlinearity)
        self.downsampling = nn.AvgPool3d(2)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.downsampling(x)
        return x


def activation(nonlinearity):
    return NONLINEARITY_DICT[nonlinearity]


class FromRGB(nn.Sequential):
    def __init__(self, channels_in, filters, nonlinearity, param=None):
        super(FromRGB, self).__init__()
        self.fromrgb = nn.Sequential(
            EqualizedConv3d(in_channels=channels_in,
                            out_channels=filters,
                            kernel_size=1,
                            nonlinearity=nonlinearity,
                            param=param),
            activation(nonlinearity)
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
    
class PrintShape(nn.Module):
    def __init__(self, comment=None):
        super(PrintShape, self).__init__()
        self.comment = comment

    def forward(self, input):
        print(self.comment, input.shape, input.sum())
        print('var_mean', torch.var_mean(input))
        return input


class Discriminator(nn.Module):
    def __init__(self, phase, num_phases, base_dim, latent_dim, base_shape, nonlinearity,
                 param=None):
        super(Discriminator, self).__init__()
        self.channels = base_shape[0]
        self.base_shape = base_shape[1:]
        self.phase = phase
        self.num_phases = num_phases
        self.base_dim = base_dim
        self.nonlinearity = nonlinearity
        if nonlinearity == 'leaky_relu':
            assert param is not None

        self.param = param

        filters_in = num_filters(phase, num_phases, base_dim)
        filters_out = num_filters(phase - 1, num_phases, base_dim)
        self.fromrgb_current = FromRGB(self.channels, filters_in, nonlinearity, param=param)
        self.fromrgb_prev = FromRGB(self.channels, filters_out, nonlinearity, param=param) if \
            self.phase > 1 \
            else None
        
        self.blocks = nn.ModuleDict()
        for i in range(2, phase + 1):
            filters_in = num_filters(i, num_phases, base_dim)
            filters_out = num_filters(i - 1, num_phases, base_dim)
            self.blocks[f'block_phase_{i}'] = DiscriminatorBlock(filters_in, filters_out,
                                                                 nonlinearity, param)
        
        self.downscale = nn.AvgPool3d(2)
            
        self.discriminator_out = nn.Sequential(
            # PrintShape(comment='out_input'),
            # MinibatchStandardDeviation(),
            # PrintShape(comment='minibatchstddev'),
            EqualizedConv3d(base_dim, base_dim, 3, padding=1, nonlinearity=nonlinearity,
                            param=param),
            activation(self.nonlinearity),
            # PrintShape(comment='conv3d'),
            nn.Flatten(),
            EqualizedLinear(np.product(self.base_shape) * base_dim, latent_dim,
                            nonlinearity=nonlinearity, param=param),
            activation(self.nonlinearity),
            # PrintShape(comment='linear'),
            EqualizedLinear(latent_dim, 1, nonlinearity='linear'),
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def grow(self):
        self.phase += 1

        filters_in = num_filters(self.phase, self.num_phases, self.base_dim)
        filters_out = num_filters(self.phase - 1, self.num_phases, self.base_dim)

        self.blocks[f'block_phase_{self.phase}'] = DiscriminatorBlock(filters_in, filters_out,
                                                                      self.nonlinearity,
                                                                      param=self.param)

        self.fromrgb_prev = self.fromrgb_current
        self.fromrgb_current = FromRGB(self.channels, filters_in, self.nonlinearity,
                                       param=self.param)
        
        self.to(self.device)

    
    def forward(self, input, alpha):
        
        input = input.to(self.device)

        x_downscale = input.clone()
        x = self.fromrgb_current(input)

        for i in reversed(range(2, self.phase + 1)):
            x = self.blocks[f'block_phase_{i}'](x)

            if i == self.phase:
                fromrgb_prev = self.fromrgb_prev(self.downscale(x_downscale))
                x = alpha * fromrgb_prev + (1 - alpha) * x

        x = self.discriminator_out(x)
        return x
            
                    
        
class ChannelNormalization(nn.Module):
    def __init__(self):
        super(ChannelNormalization, self).__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

    
class GeneratorBlock(nn.Sequential):
    def __init__(self, filters_in, filters_out, nonlinearity, param=None):
        super(GeneratorBlock, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv1 = EqualizedConv3d(filters_in, filters_out, 3, nonlinearity, padding=1,
                                     param=param)
        self.conv2 = EqualizedConv3d(filters_out, filters_out, 3, nonlinearity, padding=1,
                                     param=param)
        self.act = activation(nonlinearity)
        self.cn = ChannelNormalization()
    
    def forward(self, input):
        x = self.upsampling(input)
        x = self.conv1(x)
        x = self.act(x)
        x = self.cn(x)
        x = self.conv2(x)
        x = self.cn(x)
        x = self.act(x)
        return x


class ToRGB(nn.Sequential):
    def __init__(self, filters_in, channels):
        super(ToRGB, self).__init__()
        self.conv = EqualizedConv3d(filters_in, channels, 1, nonlinearity='linear')
        
    def forward(self, input):
        return self.conv(input)
        
        
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return torch.reshape(input, self.shape)
    
    
class Generator(nn.Module):
    def __init__(self, phase, num_phases, base_dim, latent_dim, base_shape,
                 nonlinearity, param=None):
        super(Generator, self).__init__()
        
        self.channels = base_shape[0]
        self.base_shape = base_shape[1:]
        self.phase = phase
        self.latent_dim = latent_dim
        self.base_dim = base_dim
        self.num_phases = num_phases
        self.nonlinearity = nonlinearity
        self.param = param
        if nonlinearity == 'leaky_relu':
            assert param is not None
        
        self.generator_in = nn.Sequential(
            EqualizedLinear(latent_dim, np.product(self.base_shape) * base_dim,
                            nonlinearity=nonlinearity, param=param),
            activation(nonlinearity),
            Reshape([-1, base_dim] + list(self.base_shape)),
            EqualizedConv3d(base_dim, base_dim, 3, padding=1, nonlinearity=nonlinearity,
                            param=param),
            activation(nonlinearity),
            ChannelNormalization(),
        )
                
        filters_in = num_filters(phase - 1, num_phases, base_dim)
        filters_out = num_filters(phase, num_phases, base_dim)
        
        self.torgb_current = ToRGB(filters_out, self.channels)
        self.torgb_prev = ToRGB(filters_in, self.channels) if phase > 1 else None
        
        self.blocks = nn.ModuleDict()

        for i in range(2, phase + 1):
            filters_in = num_filters(i - 1, num_phases, base_dim)
            filters_out = num_filters(i, num_phases, base_dim)
            self.blocks[f'block_phase_{i}'] = GeneratorBlock(filters_in, filters_out,
                                                             nonlinearity, param)
        
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def grow(self):
        self.phase += 1
        filters_in = num_filters(self.phase - 1, self.num_phases, self.base_dim)
        filters_out = num_filters(self.phase, self.num_phases, self.base_dim)
        self.blocks[f'block_phase_{self.phase}'] = GeneratorBlock(filters_in, filters_out,
                                                                  self.nonlinearity,
                                                                  param=self.param)
        self.torgb_prev = self.torgb_current
        self.torgb_current = ToRGB(filters_out, self.channels)
        
        self.to(self.device)
                
    def forward(self, input, alpha):
        input = input.to(self.device)
        # print('input', torch.var_mean(input))
        x = self.generator_in(input)

        # print('gen_in', torch.var_mean(x))

        x_upsample = None
        for i in range(2, self.phase + 1):
            
            if i == self.phase:
                x_upsample = self.upsample(self.torgb_prev(x))

            x = self.blocks[f'block_phase_{i}'](x)
            # print(f'x_{i}', torch.var_mean(x))

        images_out = self.torgb_current(x)
        
        if x_upsample is not None:
            images_out = alpha * x_upsample + (1 - alpha) * images_out

        return images_out


if __name__ == '__main__':


    num_phases = 8
    base_shape = (1, 1, 4, 4)


    def test_blocks():

        d_block = DiscriminatorBlock(128, 32, nonlinearity='leaky_relu',
                                     param=LEAKINESS)

        g_block = GeneratorBlock(128, 32, nonlinearity='leaky_relu', param=LEAKINESS)
        x_in = torch.randn(64, 128, 4, 4, 4)
        print('var mean in:', torch.var_mean(x_in))

        x_out = d_block(x_in)
        print('var mean out', torch.var_mean(x_out))

        x_out = g_block(x_in)
        print('var mean out', torch.var_mean(x_out))

        x_in = torch.randn(64, 512, 512)
        print('var mean in:', torch.var_mean(x_in))

        linear = EqualizedLinear(512, 512, nonlinearity='leaky_relu', param=LEAKINESS)
        x_out = nn.functional.leaky_relu(linear(x_in), negative_slope=LEAKINESS)
        print('var mean out', torch.var_mean(x_out))


    def test_growing():
        discriminator = Discriminator(1, num_phases, 256, 256, base_shape,
                                      nonlinearity='leaky_relu', param=LEAKINESS)
        generator = Generator(1, num_phases, 256, 256, base_shape, nonlinearity='leaky_relu',
                              param=LEAKINESS)

        for phase in range(1, 6): # To avoid memory overload.

            if phase > 1:
                discriminator.grow()
                generator.grow()
            print('generator parametrs:', sum(p.numel() for p in generator.parameters()))
            print('discriminator parametrs:', sum(p.numel() for p in discriminator.parameters()))
            x_in = torch.randn([1] + [base_shape[0]] + list(np.array(base_shape)[1:] * 2 ** (phase - 1)))
            x_out = discriminator(x_in, alpha=0.5)

            del x_in, x_out

            x_in = torch.randn(1, 256)
            x_out = generator(x_in, alpha=0.5)
            print(x_out.shape)

            del x_in, x_out


    def test_starting():
        for phase in range(1, 6): # To avoid memory overload.

            discriminator = Discriminator(phase, num_phases, 256, 256, base_shape,
                                          use_swish=False, )
            print('discriminator parametrs:', sum(p.numel() for p in discriminator.parameters()))

            x_in = torch.randn([1] + [base_shape[0]] + list(np.array(base_shape)[1:] * 2 ** (phase - 1)))
            x_out = discriminator(x_in, alpha=0)
            del x_in, x_out, discriminator

            generator = Generator(phase, num_phases, 256, 256, base_shape, use_swish=False)
            print('generator parametrs:', sum(p.numel() for p in generator.parameters()))
            x_in = torch.randn(1, 256)
            x_out = generator(x_in, alpha=0.5)
            print(x_out.shape)

        del x_in, x_out, generator


    test_blocks()
    test_growing()

