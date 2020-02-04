import torch

def wasserstein_loss(y_pred):
    return y_pred.mean()


def compute_gradient_penalty(discriminator, real_samples, 
                             fake_samples, alpha, gradient_penalty_weight=10):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    random_uniform = torch.rand(real_samples.shape[0], 1, 1, 1, 1).to(real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (random_uniform * real_samples + ((1 - random_uniform) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, alpha)
    fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(real_samples.device), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty * gradient_penalty_weight
