import torch
import numpy as np
from loss import wasserstein_loss, compute_gradient_penalty
from utils import write_summary
import matplotlib.pyplot as plt
import horovod.torch as hvd



def train(generator, discriminator, g_optim, d_optim, data_loader,
          mixing_epochs, stabilizing_epochs, phase, writer, horovod=False):
    
    alpha = 1  # Mixing parameter.
    for epoch in range(mixing_epochs):
        if horovod:
            hvd.broadcast_parameters(generator.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(g_optim, root_rank=0)
            hvd.broadcast_parameters(discriminator.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(d_optim, root_rank=0)
            data_loader.sampler.set_epoch(epoch)

        x_fake, x_real, *scalars = train_epoch(data_loader, 
                             generator, discriminator, g_optim, d_optim, alpha)
        
        # Tensorboard
        g_lr = g_optim.param_groups[0]['lr']
        d_lr = d_optim.param_groups[0]['lr']
        scalars = list(scalars) + [epoch, alpha, g_lr, d_lr]
        global_step = (phase - 1) * (mixing_epochs + stabilizing_epochs) + epoch
        if writer:
            write_summary(writer, global_step, x_real, x_fake, scalars)
        
        # Update alpha
        alpha -= 1 / mixing_epochs
        assert alpha >= -1e-4, alpha
        
    alpha = 0
    for epoch in range(mixing_epochs, mixing_epochs + stabilizing_epochs):
        if horovod:
            hvd.broadcast_parameters(generator.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(g_optim, root_rank=0)
            hvd.broadcast_parameters(discriminator.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(d_optim, root_rank=0)
            data_loader.sampler.set_epoch(epoch)
        x_fake, x_real, *scalars = train_epoch(data_loader, 
                             generator, discriminator, g_optim, d_optim, alpha)
        
        # Tensorboard
        g_lr = g_optim.param_groups[0]['lr']
        d_lr = d_optim.param_groups[0]['lr']
        scalars = list(scalars) + [epoch, alpha, g_lr, d_lr]
        global_step = (phase - 1) * (mixing_epochs + stabilizing_epochs) + epoch
        if writer:
            write_summary(writer, global_step, x_real, x_fake, scalars)


def train_epoch(data_loader, generator, discriminator, generator_optim, discriminator_optim, alpha):
    
    d_losses = []
    g_losses = []
    distances = []
    
    for i, x_real in enumerate(data_loader):
        
        # Train discriminator.
        generator.eval()
        for p in generator.parameters():
            p.requires_grad = False
 
        discriminator.train()
        for p in discriminator.parameters():
            p.requires_grad = True
        
        x_real = x_real.to(discriminator.device)
        z = torch.randn(x_real.shape[0], generator.latent_dim)
        x_fake = generator(z, alpha)
        
        d_real = discriminator(x_real, alpha)
        d_fake = discriminator(x_fake, alpha)
                
        gp_loss = compute_gradient_penalty(discriminator, x_real, x_fake, alpha)
        real_loss = wasserstein_loss(d_real)
        fake_loss = wasserstein_loss(d_fake)
        
        d_loss = -real_loss + fake_loss + gp_loss
        
        discriminator_optim.zero_grad()
        d_loss.backward()
        discriminator_optim.step()
        
        d_losses.append(d_loss.item())
        
        del x_real, z, x_fake, d_fake, gp_loss, real_loss, fake_loss, d_loss
        
        # Train generator.
        generator.train()
        for p in generator.parameters():
            p.requires_grad = True
            
        discriminator.eval()
        for p in discriminator.parameters():
            p.requires_grad = False
        
        z = torch.randn(x_real.shape[0], generator.latent_dim)
        x_fake = generator(z, alpha)
        d_fake = discriminator(x_fake, alpha)
        g_loss = -wasserstein_loss(d_fake)
        
        generator_optim.zero_grad()
        g_loss.backward()
        generator_optim.step()
        
        g_losses.append(g_loss.item())
        distances.append(d_real.mean().item() - d_fake.mean().item())
        
        del z, x_fake, d_fake, g_loss

    return x_fake[0].detach().cpu(), x_real[0].cpu(), np.mean(d_losses), np.mean(g_losses), np.mean(distances)
        