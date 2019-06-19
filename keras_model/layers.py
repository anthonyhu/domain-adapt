import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Add, Lambda

from functools import partial


def conv_block(x, filters, kernel_size=3, strides=1, norm='none', activation='none', transposed=False, bias=False):
    """ Conv, BN and LeakyReLU"""
    if norm == 'bn':
        norm = BatchNormalization()
    elif norm == 'in':
        norm = Lambda(instance_norm2d)
    elif norm == 'none':
        norm = None
    else:
        raise ValueError('Not recognised norm: {}'.format(norm))

    if activation == 'relu':
        activation = tf.keras.activations.relu
    elif activation == 'lrelu':
        activation = leaky_relu
    elif activation == 'tanh':
        activation = tf.keras.activations.tanh
    elif activation == 'none':
        activation = None
    else:
        raise ValueError('Not recognised activation: {}'.format(activation))

    if transposed:
        Conv_ = Conv2DTranspose
    else:
        Conv_ = Conv2D

    x = Conv_(filters, kernel_size, strides, padding='same', kernel_initializer='he_normal', use_bias=bias)(x)

    if norm:
        x = norm(x)
    if activation:
        x = Lambda(activation)(x)
    return x


def res_block(x, filters, kernel_size=3, strides=1, norm='in', activation='relu', last_block=False, bias=False):
    """ Basic residual block."""
    if activation == 'relu':
        second_activ = tf.keras.activations.relu
    elif activation == 'lrelu':
        second_activ = leaky_relu
    elif activation == 'tanh':
        second_activ = tf.keras.activations.tanh
    elif activation == 'none':
        second_activ = None
    else:
        raise ValueError('Not recognised activation: {}'.format(activation))
    identity = x

    x = conv_block(x, filters, kernel_size, strides, norm=norm, activation=activation, bias=bias)

    if last_block:
        norm = 'none'
        bias = True
        second_activ = None
    x = conv_block(x, filters, kernel_size, strides, norm=norm, activation='none', bias=bias)

    x = Add()([identity, x])

    if second_activ:
        x = Lambda(second_activ)(x)
    return x


def sample(mean):
    noise = K.random_normal(K.shape(mean))
    return mean + noise


# This function is a workaround because ReLU layer cannot be loaded atm (https://github.com/tensorflow/tensorflow/issues/22697)
def leaky_relu(x, alpha=0.2):
    return tf.keras.activations.relu(x, alpha=alpha)


def instance_norm2d(x, epsilon=1e-5):
    """ Input (batch_size, H, W, C)"""
    mean = K.mean(x, (1, 2), keepdims=True)
    stddev = K.std(x, (1, 2), keepdims=True)
    return (x - mean) / (stddev + epsilon)
