import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Lambda, Conv2DTranspose, AveragePooling2D, Flatten, Dense, UpSampling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from glob import glob

from .layers import conv_block, res_block, sample, instance_norm2d
from .losses import reconst_loss, kl_div_loss, compute_vgg_loss
from .dataset import load_day_and_night, create_dataset, load_batch
from .utils import convert_to_uint8, preprocess_vgg, write_log

lr_rate = 1e-4
lambda_0 = 1     # GAN coefficient
lambda_1 = 10    # Reconstruction coefficient
lambda_2 = 0.01  # KL divergence coefficient
lambda_3 = 0     # perceptual loss


def build_encoder(input_size, norm='in', activation='relu', name=''):
    """
    Parameters
    ----------
        input_size: tuple(int, int, int)
        name: str

    Returns
    -------
        encoder: tf.keras.model.Model
            input: image
            outputs: z_mean and z (a sample)
    """
    inputs = Input(input_size)
    h = conv_block(inputs, 64, 7, 1, norm, activation)
    h = conv_block(h, 128, 3, 2, norm, activation)
    h = conv_block(h, 256, 3, 2, norm, activation)
    h = res_block(h, 256, 3, 1, norm, activation)
    h = res_block(h, 256, 3, 1, norm, activation)
    h = res_block(h, 256, 3, 1, norm, activation)
    h = res_block(h, 256, 3, 1, norm, activation, last_block=True)
    z_mean = h
    z = Lambda(sample)(z_mean)
    encoder = Model(inputs=inputs, outputs=[z_mean, z], name=name)
    return encoder


def build_decoder(latent_size, norm='in', activation='relu', name=''):
    """
    Parameters
    ----------
        latent_size: tuple(int, int, int)
        name: str

    Returns
    -------
        decoder: tf.keras.model.Model
            input: latent tensor
            output: reconstructed image
    """
    latent_inputs = Input(latent_size)
    h = latent_inputs
    h = res_block(h, 256, 3, 1, norm, activation)
    h = res_block(h, 256, 3, 1, norm, activation)
    h = res_block(h, 256, 3, 1, norm, activation)
    h = res_block(h, 256, 3, 1, norm, activation)
    h = UpSampling2D(2)(h)
    h = conv_block(h, 128, 5, 1, norm, activation)
    h = UpSampling2D(2)(h)
    h = conv_block(h, 64, 5, 1, norm, activation)
    outputs = conv_block(h, 3, 7, 1, norm='none', activation='tanh', bias=True)

    decoder = Model(inputs=latent_inputs, outputs=outputs, name=name)
    return decoder

# TODO: make discriminator multi-scale
def build_discriminator(input_size, norm='none', activation='lrelu', name=''):
    """
    Parameters
    ----------
        input_size: tuple(int, int, int)
        name: str

    Returns
    -------
        discriminator: tf.keras.model.Model
            input: image
            outputs: soft labels
    """
    inputs = Input(input_size)
    h = conv_block(inputs, 64, 3, 2, norm, activation)
    h = conv_block(h, 128, 3, 2, norm, activation)
    h = conv_block(h, 256, 3, 2, norm, activation)
    h = conv_block(h, 512, 3, 2, norm, activation)
    h = conv_block(h, 1, 1, 1, norm='none', activation='none')
    outputs = Flatten()(h)

    discriminator = Model(inputs=inputs, outputs=outputs, name=name)
    discriminator.summary()
    return discriminator


def build_vgg16(image_size, name):
    """ vgg16 preprocessing -> vgg16 model -> instance norm
        Input: Image in [-1, 1]
    """
    vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                              input_tensor=None, input_shape=(*image_size, 3))

    inputs = tf.keras.layers.Input((*image_size, 3))
    x = tf.keras.layers.Lambda(preprocess_vgg)(inputs)
    x = vgg16(x)
    inst_norm_x = tf.keras.layers.Lambda(instance_norm2d)(x)
    vgg16_model = tf.keras.models.Model(inputs, inst_norm_x, name=name)
    vgg16_model.trainable = False
    return vgg16_model


def build_controller(input_size, name):
    inputs = Input(input_size)
    h = conv_block(inputs, 256, 3, 2)
    h = conv_block(h, 128, 3, 2)
    h = conv_block(h, 64, 3, 2)
    h = conv_block(h, 32, 3, 2)
    h = AveragePooling2D(pool_size=(3, 5))(h)
    h = Flatten()(h)
    h = Dense(16)(h)
    outputs = Dense(1)(h)

    controller = Model(inputs=inputs, outputs=outputs, name=name)
    controller.summary()
    return controller


class CycleVAE():
    """ UNIT network (Unsupervised Image-to-Image Translation)"""
    def __init__(self, save_folder, image_size=(512, 512), checkpoint_path=''):
        self.save_folder = save_folder
        self.image_size = image_size
        self.checkpoint_path = checkpoint_path
        self._build_model()

        if self.checkpoint_path:
            print('Loading weights from checkpoint {}'.format(os.path.basename(self.checkpoint_path)))
            self.cycle_vae.load_weights(self.checkpoint_path)
        else:
            checkpoint = glob(os.path.join(self.save_folder, '*.h5'))
            if checkpoint:
                print('Loading weights from checkpoint {}'.format(os.path.basename(checkpoint[0])))
                self.cycle_vae.load_weights(checkpoint[0])

        self.X_day_val, self.X_night_val = None, None

    def _build_model(self):
        """
        Creates
        -------
            self.cycle_vae: keras Model
                full model

        """
        K.clear_session()
        optimizer = Adam(lr=lr_rate, beta_1=0.5, beta_2=0.999)

        # Source distribution
        self.E_source = build_encoder((*self.image_size, 3),  name='E_source')
        latent_size = self.E_source.outputs[1].get_shape()[1:]
        self.G_source = build_decoder(latent_size, name='G_source')

        # Target distribution
        self.E_targ = build_encoder((*self.image_size, 3),  name='E_targ')
        latent_size = self.E_targ.outputs[1].get_shape()[1:]
        self.G_targ = build_decoder(latent_size, name='G_targ')

        # VGG16 feature extractor
        #self.vgg16 = build_vgg16(self.image_size, 'vgg16')
        #self.vgg16.summary()

        self.x_source = self.E_source.input
        self.x_targ = self.E_targ.input

        # Source reconstruction
        self.source_reconst = self.G_source(self.E_source(self.x_source)[1])
        # Target reconstruction
        self.targ_reconst = self.G_targ(self.E_targ(self.x_targ)[1])

        # Translations
        self.translation_to_source = self.G_source(self.E_targ(self.x_targ)[1])
        self.translation_to_targ = self.G_targ(self.E_source(self.x_source)[1])

        # Cycle reconst: source -> targ -> source
        self.cycle1_reconst = self.G_source(self.E_targ(self.translation_to_targ)[1])
        # Cycle reconst: targ -> source -> targ
        self.cycle2_reconst = self.G_targ(self.E_source(self.translation_to_source)[1])


        # GANs
        # Build and compile discriminators
        self.D_source = build_discriminator((*self.image_size, 3), name='D_source')
        self.D_targ = build_discriminator((*self.image_size, 3), name='D_targ')
        self.D_source.compile(optimizer=optimizer, loss='mse', loss_weights=[lambda_0], metrics=['binary_accuracy'])
        self.D_targ.compile(optimizer=optimizer, loss='mse', loss_weights=[lambda_0], metrics=['binary_accuracy'])

        # set discriminator weights to False
        self.D_source.trainable = False
        self.D_targ.trainable = False

        valid_source = self.D_source(self.translation_to_source)
        valid_targ = self.D_targ(self.translation_to_targ)

        inputs = [self.x_source, self.x_targ]
        outputs = [valid_source, valid_targ]

        self.cycle_vae = Model(inputs=inputs, outputs=outputs, name='cycle_vae')

        def in_domain_reconst_loss(y_true, y_pred):
            r_source_loss = lambda_1 * reconst_loss(self.x_source, self.source_reconst)
            r_targ_loss = lambda_1 * reconst_loss(self.x_targ, self.targ_reconst)
            return r_source_loss + r_targ_loss

        def in_domain_kl_loss(y_true, y_pred):
            kl_source_loss = lambda_2 * kl_div_loss(self.E_source.outputs[0])
            kl_targ_loss = lambda_2 * kl_div_loss(self.E_targ.outputs[0])
            return kl_source_loss + kl_targ_loss

        def cyclic_loss(y_true, y_pred):
            cyclic_1_loss = lambda_1 * reconst_loss(self.x_source, self.cycle1_reconst)
            cyclic_2_loss = lambda_1 * reconst_loss(self.x_targ, self.cycle2_reconst)
            return cyclic_1_loss + cyclic_2_loss

        def kl_cyclic_loss(y_true, y_pred):
            kl_cyclic_1_loss = lambda_2 * kl_div_loss(self.E_source(self.translation_to_source)[0])
            kl_cyclic_2_loss = lambda_2 * kl_div_loss(self.E_targ(self.translation_to_targ)[0])
            return kl_cyclic_1_loss + kl_cyclic_2_loss

        def gan_loss(y_true, y_pred):
            gan_loss1 = lambda_0 * K.mean(K.square(y_true - self.D_source(self.translation_to_source)))
            gan_loss2 = lambda_0 * K.mean(K.square(y_true - self.D_targ(self.translation_to_targ)))
            return gan_loss1 + gan_loss2

        def vgg_loss(y_true, y_pred):
            vgg_loss1 = lambda_3 * compute_vgg_loss(self.vgg16(self.x_source), self.vgg16(self.translation_to_targ))
            vgg_loss2 = lambda_3 * compute_vgg_loss(self.vgg16(self.x_targ), self.vgg16(self.translation_to_source))
            return vgg_loss1 + vgg_loss2

        # Loss function
        def cycle_vae_loss(y_true, y_pred):
            """
            Returns
            -------
                cycle_vae_loss: tf.Tensor
                    L2 distance + KL divergence
            """
            # In-domain reconst loss
            # In-domain KL loss
            # Cyclic loss
            # Cyclic KL loss
            # GAN loss
            # Perceptual loss
            total_loss = (in_domain_reconst_loss(y_true, y_pred)
                          + in_domain_kl_loss(y_true, y_pred)
                          + cyclic_loss(y_true, y_pred)
                          + kl_cyclic_loss(y_true, y_pred)
                          + gan_loss(y_true, y_pred)
                          #+ vgg_loss(y_true, y_pred)
                         )
            return total_loss

        def dummy_loss(y_true, y_pred):
            return K.zeros(1)

        self.cycle_vae.compile(optimizer, loss=[cycle_vae_loss, dummy_loss],
                                   metrics=[in_domain_reconst_loss, in_domain_kl_loss, cyclic_loss,
                                            kl_cyclic_loss, gan_loss, dummy_loss])#, vgg_loss])
        self.cycle_vae.summary()

    def train(self, epochs=10, batch_size=1, print_interval=50):
        # Tensorboard callback
        train_callback = TensorBoard(os.path.join(self.save_folder, 'train'))
        train_callback.set_model(self.cycle_vae)
        val_callback = TensorBoard(os.path.join(self.save_folder, 'val'))
        val_callback.set_model(self.cycle_vae)
        metrics_names = ['discri_loss_source', 'discri_accuracy_source', 'discri_loss_targ', 'discri_accuracy_targ',
                         'g_total_loss', 'g_reconst', 'g_kl', 'g_cyclic_reconst', 'g_cyclic_kl', 'g_gan', 'g_vgg']

        # Dataset
        X_day_train, X_night_train = load_day_and_night('train')
        self.X_day_val, self.X_night_val = load_day_and_night('val')

        n_train = len(X_day_train)
        n_val = len(self.X_day_val)
        print('Training examples: {}'.format(n_train))
        print('Validation examples: {}'.format(n_val))

        discri_dim = self.D_source.output.get_shape()[1]
        valid = np.ones((batch_size, discri_dim))
        fake = np.zeros((batch_size, discri_dim))

        val_loss_min = float('inf')
        iter_nb = 0
        for e in range(epochs):
            n_iter = n_train // batch_size
            indices = np.random.permutation(n_train)

            for i in range(n_iter):
                idx = np.random.choice(indices, size=batch_size, replace=False)
                #indices[i*batch_size : (i+1)*batch_size]
                source_batch = load_batch(X_day_train[idx], self.image_size)
                targ_batch = load_batch(X_night_train[idx], self.image_size)

                # Discriminator
                d_loss_source, d_loss_targ = self.discri_train_on_batch(source_batch, targ_batch, valid, fake)

                # Generator
                g_loss = self.cycle_vae.train_on_batch([source_batch, targ_batch], [valid, valid])


                if i%print_interval == 0:
                    print('Iteration {:4d}: Training loss:'.format(i))

                    logs = self.get_logs(d_loss_source, d_loss_targ, g_loss)
                    print('Discri source loss: {:.2f}, accuracy: {:.1f}%'.format(logs[0], logs[1]))
                    print('Discri targ loss: {:.2f}, accuracy: {:.1f}%'.format(logs[2], logs[3]))
                    print('Gen loss: {:.2f}'.format(logs[4]))
                    print('Reconst: {:.2f}, kl: {:.2f}, cyclic_reconst: {:.2f}, cyclic_kl: {:.2f},gan: {:.2f}, vgg: {:.2f}'\
                          .format(*logs[5:]))

                    write_log(train_callback, metrics_names, logs, iter_nb)

                if i%100 == 0:
                    self.predict(e + 1, iter=iter_nb, save_fig=True)
                    print('Figure saved.')

                iter_nb += 1

            # Calculate validation loss
            val_loss = 0.0
            n_iter_val = n_val // batch_size
            indices = np.arange(n_val)
            logs = np.zeros(len(metrics_names))
            for i in range(n_iter_val):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                source_batch = load_batch(self.X_day_val[idx], self.image_size)
                targ_batch = load_batch(self.X_night_val[idx], self.image_size)


                # Discriminator
                d_loss_source, d_loss_targ = self.discri_train_on_batch(source_batch, targ_batch, valid, fake)

                # Generator
                g_loss = self.cycle_vae.train_on_batch([source_batch, targ_batch], [valid, valid])
                logs += self.get_logs(d_loss_source, d_loss_targ, g_loss)

                val_loss += g_loss[0]

            val_loss /= n_iter_val
            logs /= n_iter_val

            print('\n Epoch {} - Validation loss: {:.2f}'.format(e + 1, val_loss))
            print('Discri source loss: {:.2f}, accuracy: {:.1f}%'.format(logs[0], logs[1]))
            print('Discri targ loss: {:.2f}, accuracy: {:.1f}%'.format(logs[2], logs[3]))
            print('Gen loss: {:.2f}'.format(logs[4]))
            print('Reconst: {:.2f}, kl: {:.2f}, cyclic_reconst: {:.2f}, cyclic_kl: {:.2f},gan: {:.2f}, vgg: {:.2f}' \
                  .format(*logs[5:]))

            write_log(val_callback, metrics_names, logs, iter_nb)

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                print('Saving model\n')
                weights_filename = os.path.join(self.save_folder, 'cycle_vae.epoch{:02d}-val_loss{:.2f}.h5'.format(e+1, val_loss))
                self.cycle_vae.save_weights(weights_filename)

    def predict(self, epoch=0, save_fig=True, iter=0, n_examples=3):
        if self.X_day_val is None or self.X_night_val is None:
            self.X_day_val, self.X_night_val = load_day_and_night('val')
        f_source_to_targ = K.function([self.x_source, K.learning_phase()], [self.G_targ(self.E_source(self.x_source)[0])])
        # TODO: might need to use the mus instead of the zs
        f_cycle1_reconst = K.function([self.x_source, K.learning_phase()], [self.cycle1_reconst])

        for i, filename in enumerate(np.random.choice(self.X_day_val, n_examples)):
            img = load_batch([filename], self.image_size)

            x_targ = f_source_to_targ([img, 0])[0]
            x_cycle1_reconst = f_cycle1_reconst([img, 0])[0]

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(convert_to_uint8(img)[0])
            plt.title('Original')
            plt.subplot(132)
            plt.imshow(convert_to_uint8(x_targ[0]))
            plt.title('Translated image')
            plt.subplot(133)
            plt.imshow(convert_to_uint8(x_cycle1_reconst[0]))
            plt.title('Cycle reconstruction')
            fig_filename = os.path.join(self.save_folder, 'epoch{:02d}-iter{:04d}-day-example{}.png'.format(epoch, iter, i))
            if save_fig:
                plt.savefig(fig_filename)
                plt.close()
            else:
                plt.show()

        f_targ_to_source = K.function([self.x_targ, K.learning_phase()], [self.G_source(self.E_targ(self.x_targ)[0])])
        # TODO: might need to use the mus instead of the zs
        f_cycle2_reconst = K.function([self.x_targ, K.learning_phase()], [self.cycle2_reconst])

        for i, filename in enumerate(np.random.choice(self.X_night_val, n_examples)):
            img = load_batch([filename], self.image_size)

            x_source = f_targ_to_source([img, 0])[0]
            x_cycle2_reconst = f_cycle2_reconst([img, 0])[0]

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(convert_to_uint8(img)[0])
            plt.title('Original')
            plt.subplot(132)
            plt.imshow(convert_to_uint8(x_source[0]))
            plt.title('Translated image')
            plt.subplot(133)
            plt.imshow(convert_to_uint8(x_cycle2_reconst[0]))
            plt.title('Cycle reconstruction')
            fig_filename = os.path.join(self.save_folder, 'epoch{:02d}-iter{:04d}-night-example{}.png'.format(epoch, iter, i))
            if save_fig:
                plt.savefig(fig_filename)
                plt.close()
            else:
                plt.show()


    def discri_train_on_batch(self, source_batch, targ_batch, valid, fake):
        gen_translated_to_source = self.G_source.predict(self.E_targ.predict(targ_batch)[1])
        gen_translated_to_targ = self.G_targ.predict(self.E_source.predict(source_batch)[1])

        d_loss_source_real = self.D_source.train_on_batch(source_batch, valid)
        d_loss_source_fake = self.D_source.train_on_batch(gen_translated_to_source, fake)
        d_loss_source = 0.5 * np.add(d_loss_source_real, d_loss_source_fake)

        d_loss_targ_real = self.D_targ.train_on_batch(targ_batch, valid)
        d_loss_targ_fake = self.D_targ.train_on_batch(gen_translated_to_targ, fake)
        d_loss_targ = 0.5 * np.add(d_loss_targ_real, d_loss_targ_fake)

        return d_loss_source, d_loss_targ

    def get_logs(self, d_loss_source, d_loss_targ, g_loss):
        logs = np.array([d_loss_source[0], 100 * d_loss_source[1], d_loss_targ[0], 100 * d_loss_targ[1],
                         g_loss[0], g_loss[3], g_loss[4], g_loss[5], g_loss[6], g_loss[7], g_loss[8]])
        return logs


def vae_model(image_size=(180, 320)):
    """ Compiles a VAE model (architecture from https://arxiv.org/pdf/1703.00848.pdf)
    base vae from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    """
    K.clear_session()

    # Encoder
    encoder = build_encoder((*image_size, 3),  name='encoder')

    latent_size = encoder.outputs[1].get_shape()[1:]
    decoder = build_decoder(latent_size, name='decoder')

    inputs = encoder.input
    outputs = decoder(encoder(inputs)[1])
    vae = Model(inputs=inputs, outputs=outputs, name='vae')

    # Loss function
    def vae_loss(y_true, y_pred):
        """
        Parameters
        ----------
            inputs: tf.Tensor (-1, h, w, 3)
            outputs: tf.Tensor (-1, h, w, 3)

        Returns
        -------
            vae_loss: tf.Tensor
                L2 distance + KL divergence
        """
        r_loss = lambda_1 * reconst_loss(inputs, outputs)
        z_mean = encoder.outputs[0]
        kl_loss = lambda_2 * kl_div_loss(z_mean)

        return r_loss + kl_loss

    optimizer = Adam(lr=lr_rate, beta_1=0.5, beta_2=0.999)
    vae.compile(optimizer, loss=vae_loss)
    vae.summary()

    return vae