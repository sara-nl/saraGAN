import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch
import os
import shutil
import stat

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_summary(writer, step, x_real, x_fake, scalars):
    
    axes = [1, 0, 2, 3]
    x_fake = np.transpose(x_fake, axes)
    fake_grid = torchvision.utils.make_grid(x_fake, padding=False, normalize=True)
    writer.add_image(f'x_fake', fake_grid, step)
    
    x_real = np.transpose(x_real, axes )
    real_grid = torchvision.utils.make_grid(x_real, padding=False, normalize=True)
    writer.add_image('x_real', real_grid, step)
    
    writer.add_scalar('d_loss', scalars[0], step)
    writer.add_scalar('g_loss', scalars[1], step)
    # writer.add_scalars('loss', {'d_loss': scalars[0], 'g_loss': scalars[1]}, step)
    writer.add_scalar('distance', scalars[2], step)
    writer.add_scalar('gp', scalars[3], step)
    writer.add_scalar('epoch', scalars[4], step)
    writer.add_scalar('alpha', scalars[5], step)
    writer.add_scalar('g_lr', scalars[6], step)
    writer.add_scalar('d_lr', scalars[7], step)
    writer.add_scalar('img\/s', scalars[8], step)
    # writer.add_scalars('lr', {'g_lr': scalars[5], 'd_lr': scalars[6]}, step)
    
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
