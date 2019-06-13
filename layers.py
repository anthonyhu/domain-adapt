import torch.nn as nn

from functools import partial


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