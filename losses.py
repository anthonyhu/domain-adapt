import torch


def vae_loss(x, x_hat, mean, r_coef=100, kl_coef=0.1):
    """ Vae loss function"""
    r_loss = torch.abs(x - x_hat).mean()
    kl_loss = 0.5 * torch.pow(mean, 2).mean()
    return r_coef * r_loss + kl_coef * kl_loss