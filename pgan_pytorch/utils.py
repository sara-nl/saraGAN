import numpy as np
import torchvision
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_summary(writer, step, x_real, x_fake, scalars):
    axes = [1, 0, 2, 3]
    x_fake = np.transpose(x_fake, axes)
    x_real = np.transpose(x_real, axes )
    fake_grid = torchvision.utils.make_grid(x_fake, padding=False, normalize=True)
    real_grid = torchvision.utils.make_grid(x_real, padding=False, normalize=True)
    
    writer.add_image('x_real', real_grid, step)
    writer.add_image('x_fake', fake_grid, step)
    
    writer.add_scalar('d_loss', scalars[0], step)
    writer.add_scalar('g_loss', scalars[1], step)
    # writer.add_scalars('loss', {'d_loss': scalars[0], 'g_loss': scalars[1]}, step)
    writer.add_scalar('distance', scalars[2], step)
    writer.add_scalar('epoch', scalars[3], step)
    writer.add_scalar('alpha', scalars[4], step)
    writer.add_scalar('g_lr', scalars[5], step)
    writer.add_scalar('d_lr', scalars[6], step)
    writer.add_scalar('img\/s', scalars[7], step)
    # writer.add_scalars('lr', {'g_lr': scalars[5], 'd_lr': scalars[6]}, step)
    
