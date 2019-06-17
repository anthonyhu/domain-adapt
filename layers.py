import torch
import numpy as np
import torch.nn as nn

from functools import partial
from torch.autograd import Variable
import torch.nn.functional as F


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
        mean = self.model(x)
        if self.training:
            z = mean + torch.randn_like(mean).to(self.device)
        else:
            z = mean
        return z, mean


class Decoder(nn.Module):
    def __init__(self, initial_c=256, output_channels=3, n_res=4, n_conv=3, activation='relu'):
        super().__init__()
        self.model = []

        in_channels = initial_c
        for i in range(n_res):
            self.model.append(ResBlock(in_channels, norm='in', activation=activation))

        for i in range(n_conv - 1):
            self.model.append(nn.Upsample(scale_factor=2))
            self.model.append(ConvBlock(in_channels, in_channels // 2, kernel_size=5, stride=1, norm='ln', activation=activation))
            in_channels = in_channels // 2

        self.model.append(ConvBlock(in_channels, output_channels, kernel_size=7, stride=1, norm='none', activation='tanh'))

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
    def __init__(self, input_channels=3, initial_c=64, n_conv=4, norm='none', activation='lrelu'):
        super().__init__()
        self.model = []

        in_channels = input_channels
        out_channels = initial_c
        for i in range(n_conv):
            self.model.append(ConvBlock(in_channels, out_channels, 3, 2, norm=norm, activation=activation))
            in_channels = out_channels
            out_channels = 2 * in_channels

        self.model.append(ConvBlock(in_channels, 1, 1, norm='none', activation='none'))
        self.model.append(nn.AvgPool2d(3, 2, padding=1))

        self.model = nn.Sequential(*self.model)

        self.n_parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        print('Discriminator\n----------------------')
        print('Number of parameters: {}\n'.format(self.n_parameters))

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    """ Conv and optional (BN - ReLU)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm='none', activation='none', transpose=False):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=padding)

        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm =='bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
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

        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)

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

    def __init__(self, channels, norm='in', activation='lrelu', second_activation=False, last_block=False):
        super().__init__()
        self.second_activation = second_activation

        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Not recognised activation {}'.format(activation))

        self.model = []

        self.model.append(ConvBlock(channels, channels, 3, 1, norm=norm, activation=activation))
        if last_block:
            norm = 'none'
            self.activation = None
        self.model.append(ConvBlock(channels, channels, 3, 1, norm=norm, activation='none'))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        identity = x
        x = self.model(x)
        x += identity
        if self.activation and self.second_activation:
            x = self.activation(x)
        return x


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
        super().__init__()
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
        super().__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.device = params['device']
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
        return outputs  # TODO: remove [0] to keep multiscale

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                # loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
                valid = torch.ones_like(out1).to(self.device)
                fake = torch.zeros_like(out0).to(self.device)
                loss_real_a = torch.nn.MSELoss()(out1, valid)
                loss_fake_a = torch.nn.MSELoss()(out0, fake)
                loss += loss_real_a + loss_fake_a

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
                valid = torch.ones_like(out0).to(self.device)
                loss += torch.nn.MSELoss()(out0, valid)
                # loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
