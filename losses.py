import torch


def reconst_loss(x, x_hat):
    return torch.abs(x - x_hat).mean()

def kl_loss(mu):
    return 0.5 * torch.pow(mu, 2).mean()

def vae_loss(x, x_hat, mu, r_coef=100, kl_coef=0.1):
    """ Vae loss function"""
    return r_coef * reconst_loss(x, x_hat) + kl_coef * kl_loss(mu)