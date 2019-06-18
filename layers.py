import torch
import numpy as np
import torch.nn as nn

from functools import partial


class Encoder(nn.Module):
    def __init__(self, device, input_channels=3, initial_c=64, n_conv=3, n_res=4, norm='in', activation='relu'):
        super().__init__()
        self.device = device
        self.model = []

        in_channels = input_channels
        out_channels = initial_c
        for i in range(n_conv):
            kernel_size = 7 if i == 0 else 3
            stride = 1 if i == 0 else 2
            self.model.append(ConvBlock(in_channels, out_channels, kernel_size, stride, norm=norm, activation=activation))
            in_channels = out_channels
            out_channels *= 2

        for i in range(n_res):
            last_block = True if (i == (n_res-1)) else False
            self.model.append(ResBlock(in_channels, norm=norm, activation=activation, last_block=last_block))

        self.model = nn.Sequential(*self.model)

        self.n_parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        print('Encoder\n----------------------')
        print('Number of parameters: {}\n'.format(self.n_parameters))

    def forward(self, x):
        mu = self.model(x)
        if self.training:
            z = mu + torch.randn_like(mu).to(self.device)
        else:
            z = mu
        return z, mu


class Decoder(nn.Module):
    def __init__(self, initial_c=256, output_channels=3, n_res=4, n_conv=3, norm='in', activation='relu'):
        super().__init__()
        self.model = []

        in_channels = initial_c
        for i in range(n_res):
            self.model.append(ResBlock(in_channels, norm=norm, activation=activation))

        for i in range(n_conv - 1):
            self.model.append(nn.Upsample(scale_factor=2))
            self.model.append(ConvBlock(in_channels, in_channels // 2, kernel_size=5, stride=1, norm=norm, activation=activation))
            in_channels = in_channels // 2

        self.model.append(ConvBlock(in_channels, output_channels, kernel_size=7, stride=1, norm='none', activation='tanh', bias=True))

        self.model = nn.Sequential(*self.model)

        self.n_parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        print('Decoder\n----------------------')
        print('Number of parameters: {}'.format(self.n_parameters))

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """ Multiscale discriminator"""
    def __init__(self, device, n_scales=3, input_channels=3, initial_c=64, n_conv=4, norm='none', activation='lrelu'):
        super().__init__()
        self.device = device
        self.n_scales = n_scales
        self.input_channels = input_channels
        self.initial_c = initial_c
        self.n_conv = n_conv
        self.norm = norm
        self.activation = activation
        self.downsample = nn.AvgPool2d(3, 2, padding=1, count_include_pad=False)

        self.models = nn.ModuleList()

        for _ in range(self.n_scales):
            self.models.append(self._create_net())

        self.n_parameters = sum([np.prod(p.size()) for p in self.parameters()])
        print('Discriminator\n----------------------')
        print('Number of parameters: {}\n'.format(self.n_parameters))

    def _create_net(self):
        model = []
        in_channels = self.input_channels
        out_channels = self.initial_c
        for i in range(self.n_conv):
            model.append(ConvBlock(in_channels, out_channels, 3, 2, norm=self.norm, activation=self.activation, bias=True))
            in_channels = out_channels
            out_channels = 2 * in_channels

        model.append(ConvBlock(in_channels, 1, 1, norm='none', activation='none', bias=True))
        model = nn.Sequential(*model)

        return model

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
            # Downsample
            x = self.downsample(x)
        return outputs

    def discriminator_loss(self, X_real, X_fake):
        D_real_list = self.forward(X_real)
        D_fake_list = self.forward(X_fake)

        loss = 0
        D_x = 0
        D_G_x = 0
        for D_real, D_fake in zip(D_real_list, D_fake_list):
            D_x += D_real.mean()
            D_G_x += D_fake.mean()
            valid = torch.ones_like(D_real).to(self.device)
            fake = torch.zeros_like(D_fake).to(self.device)

            loss += (nn.MSELoss()(D_real, valid)
                     + nn.MSELoss()(D_fake, fake))

        D_x /= len(D_real_list)
        D_G_x /= len(D_fake_list)

        return loss, D_x, D_G_x

    def generator_loss(self, X_fake):
        D_fake_list = self.forward(X_fake)

        loss = 0
        for D_fake in D_fake_list:
            valid = torch.ones_like(D_fake).to(self.device)
            loss += nn.MSELoss()(D_fake, valid)

        return loss


class ConvBlock(nn.Module):
    """ Conv and optional (BN - ReLU)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm='none', activation='none', bias=False,
                 transpose=False):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=1)

        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm =='bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Not recognised norm {}'.format(norm))

        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Not recognised activation {}'.format(activation))

        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class ResBlock(nn.Module):
    """ Conv - BN - ReLU - Conv - BN - ADD and then ReLU
        Same number of channels in and out.
    """
    def __init__(self, channels, norm='in', activation='lrelu', bias=False, last_block=False):
        super().__init__()
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Not recognised activation {}'.format(activation))

        self.model = []

        self.model.append(ConvBlock(channels, channels, 3, 1, norm=norm, activation=activation, bias=bias))
        if last_block:
            norm = 'none'
            bias = True
            self.activation = None
        self.model.append(ConvBlock(channels, channels, 3, 1, norm=norm, activation='none', bias=bias))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        identity = x
        x = self.model(x)
        x += identity
        if self.activation:
            x = self.activation(x)
        return x
