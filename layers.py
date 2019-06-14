import torch
import numpy as np
import torch.nn as nn

from functools import partial
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, device, input_channels=3, initial_c=64, n_conv=3, n_res=4):
        super().__init__()
        self.device = device
        self.model = []

        in_channels = input_channels
        out_channels = initial_c
        for i in range(n_conv):
            kernel_size = 7 if i == 0 else 3
            stride = 1 if i == 0 else 2
            self.model.append(ConvBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels
            out_channels *= 2

        for i in range(n_res):
            last_block = True if (i == (n_res-1)) else False
            self.model.append(ResBlock(in_channels, last_block=last_block))

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
        super().__init__()
        self.model = []

        in_channels = initial_c
        for i in range(n_res):
            self.model.append(ResBlock(in_channels))

        for i in range(n_conv - 1):
            self.model.append(nn.Upsample(scale_factor=2))
            self.model.append(ConvBlock(in_channels, in_channels // 2, kernel_size=5, stride=1))
            in_channels = in_channels // 2

        self.model.append(ConvBlock(in_channels, output_channels, kernel_size=7, stride=1, activation='tanh'))

        self.model = nn.Sequential(*self.model)

        self.n_parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        print('Decoder\n----------------------')
        print('Number of parameters: {}'.format(self.n_parameters))

    def forward(self, x):
        return self.model(x)


# class Decoder_old(nn.Module):
#     def __init__(self, initial_c=256, output_channels=3, n_res=4, n_conv=3):
#         super(Decoder, self).__init__()
#         self.model = []
#
#         in_channels = initial_c
#         for i in range(n_res):
#             self.model.append(ResBlock(in_channels))
#
#         for i in range(n_conv - 1):
#             self.model.append(ConvBlock(in_channels, in_channels // 2, kernel_size=3, stride=2,
#                                         transpose=True))
#             in_channels = in_channels // 2
#
#         self.model.append(nn.ConvTranspose2d(in_channels, output_channels, 3, stride=1, padding=1))
#         self.model.append(nn.Tanh())
#
#         self.model = nn.Sequential(*self.model)
#
#         self.n_parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
#         print('Decoder\n----------------------')
#         print('Number of parameters: {}'.format(self.n_parameters))
#
#     def forward(self, x):
#         return self.model(x)


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

        self.model.append(ConvBlock(in_channels, 1, 5, norm=None, activation=None))

        self.model = nn.Sequential(*self.model)

        self.n_parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        print('Discriminator\n----------------------')
        print('Number of parameters: {}\n'.format(self.n_parameters))

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    """ Conv and optional (BN - LeakyReLU)
        No bias, slope of 0.2 in LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm='in', activation='lrelu', transpose=False):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=padding)

        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm =='bn':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)

    def forward(self, x):
        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class ResBlock(nn.Module):
    """ Conv - BN - LeakyReLU - Conv - BN - ADD and then LeakyReLU
        Same number of channels in and out.
    """

    def __init__(self, channels, norm='in', activation='lrelu', last_block=False):
        super().__init__()
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = None

        self.model = []

        self.model.append(ConvBlock(channels, channels, 3, 1, norm=norm, activation=activation))
        if last_block:
            norm = None
            self.activation = None
        self.model.append(ConvBlock(channels, channels, 3, 1, norm=norm, activation=None))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        identity = x
        x = self.model(x)
        x += identity
        if self.activation:
            x = self.activation(x)
        return x






##################################################################
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        mean = self.model(x)
        if self.training:
            z = mean + torch.randn_like(mean).to(self.device)
        else:
            z = mean
        return z, mean

class Decoder_other(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super().__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

##################################################################################
# Sequential Models
##################################################################################

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock_other(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock_other(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super().__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss