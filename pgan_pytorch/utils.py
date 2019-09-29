import numpy as np
import torchvision
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_summary(writer, global_step, x_real, x_fake, scalars):
    axes = [1, 0, 2, 3]
    x_fake = np.transpose(x_fake, axes)
    x_real = np.transpose(x_real, axes )
    fake_grid = torchvision.utils.make_grid(x_fake, padding=False, normalize=True, scale_each=True)
    real_grid = torchvision.utils.make_grid(x_real, padding=False, normalize=True, scale_each=True)
    
    writer.add_image('x_real', real_grid, global_step)
    writer.add_image('x_fake', fake_grid, global_step)
    
    scalar_names = ['d_loss', 'g_loss', 'distance', 'epoch', 'alpha', 'g_lr', 'd_lr']
    assert len(scalar_names) == len(scalars)
    
    for i, name in enumerate(scalar_names):
        writer.add_scalar(name, scalars[i], global_step)
    
    
