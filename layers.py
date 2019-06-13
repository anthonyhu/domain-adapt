import torch
import numpy as np
import torch.nn as nn

from functools import partial


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z, mean = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mean, z


class Encoder(nn.Module):
    def __init__(self, device, input_channels=3, initial_c=64, n_conv=3, n_res=4):
        super(Encoder, self).__init__()
        self.device = device
        self.model = []

        in_channels = input_channels
        out_channels = initial_c
        for i in range(n_conv ):
            kernel_size = 7 if i == 0 else 3
            stride = 1 if i == 0 else 2
            self.model.append(ConvBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels
            out_channels *= 2

        for i in range(n_res):
            last_block = True if (i == (n_res-1)) else False
            self.model.append(ResBlock(in_channels, last_block))

        self.model = nn.Sequential(*self.model)

        self.n_parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        print('Encoder\n----------------------')
        print('Number of parameters: {}\n'.format(self.n_parameters))

    def forward(self, x):
        mean = self.model(x)
        if self.training:
            z = mean + torch.randn_like(mean).to(self.device)
        else:
            z = mean
        return z, mean


class Decoder(nn.Module):
    def __init__(self, initial_c=256, output_channels=3, n_res=4, n_conv=3):
        super(Decoder, self).__init__()
        self.model = []

        in_channels = initial_c
        for i in range(n_res):
            self.model.append(ResBlock(in_channels))

        for i in range(n_conv - 1):
            self.model.append(ConvBlock(in_channels, in_channels // 2, kernel_size=3, stride=2,
                                        transpose=True))
            in_channels = in_channels // 2

        self.model.append(nn.ConvTranspose2d(in_channels, output_channels, 3, stride=1, padding=1))
        self.model.append(nn.Tanh())

        self.model = nn.Sequential(*self.model)

        self.n_parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        print('Decoder\n----------------------')
        print('Number of parameters: {}'.format(self.n_parameters))

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, initial_c=64, n_conv=6):
        super().__init__()
        self.model = []

        in_channels = input_channels
        out_channels = initial_c
        for i in range(n_conv - 1):
            self.model.append(ConvBlock(in_channels, out_channels, 3, 2))
            in_channels = out_channels
            out_channels = 2 * in_channels

        self.model.append(nn.Conv2d(in_channels, 1, 5))

        self.model = nn.Sequential(*self.model)

        self.n_parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        print('Discriminator\n----------------------')
        print('Number of parameters: {}\n'.format(self.n_parameters))

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    """ Conv - BN - LeakyReLU
        No bias, slope of 0.2 in LeakyReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, transpose=False):
        super(ConvBlock, self).__init__()
        self.model = []

        padding = int((kernel_size - 1) / 2)
        conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=padding)

        self.model.append(conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False))
        self.model.append(nn.BatchNorm2d(out_channels))
        self.model.append(nn.LeakyReLU(0.2))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    """ Conv - BN - LeakyReLU - Conv - BN - LeakyReLU
        Same number of channels in and out.
    """

    def __init__(self, channels, last_block=False):
        super(ResBlock, self).__init__()
        self.model = []

        self.model.append(ConvBlock(channels, channels, 3, 1))
        if last_block:
            self.model.append(nn.Conv2d(channels, channels, 3, 1, padding=1, bias=True))
        else:
            self.model.append(ConvBlock(channels, channels, 3, 1))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)