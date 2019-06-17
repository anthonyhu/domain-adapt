import os
import torch

import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from layers import Encoder, Decoder, Discriminator, MsImageDis
from losses import reconst_loss, kl_loss
from utils import weights_init


class UNIT(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # Tensorboard
        output_dir = params['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self.writer = SummaryWriter(output_dir, flush_secs=60)

        # Create UNIT network
        self.device = params['device']

        self.E_a = Encoder(self.device).to(self.device)
        self.E_b = Encoder(self.device).to(self.device)

        self.G_a = Decoder().to(self.device)
        self.G_b = Decoder().to(self.device)

        # self.D_a = Discriminator(self.device).to(self.device)
        # self.D_b = Discriminator(self.device).to(self.device)

        dis_params = {'dim': 64,
                      'norm': 'none',
                      'activ': 'lrelu',
                      'n_layer': 4,
                      'gan_type': 'lsgan',
                      'num_scales': 3,
                      'pad_type': 'reflect',
                      'device': self.device}
        self.D_a = MsImageDis(3, dis_params)
        self.D_b = MsImageDis(3, dis_params)

        #
        self.D_a = self.D_a.to(self.device)  # self.D_a = self.D_a.to(self.device)
        self.D_b = self.D_b.to(self.device)

        if not params['default_init']:
            print('Initialise generator with Kaiming normal and discri with Gaussian')
            self.apply(weights_init('kaiming'))
            self.D_a = self.D_a.apply(weights_init('gaussian'))
            self.D_b = self.D_b.apply(weights_init('gaussian'))

        self.generator = [self.E_a, self.E_b, self.G_a, self.G_b]
        self.discriminator = [self.D_a, self.D_b]

        # Optimizers
        betas = params['betas']
        lr = params['lr']
        D_parameters = sum([list(model.parameters()) for model in self.discriminator], [])
        self.D_optimizer = torch.optim.Adam(D_parameters, lr=lr, betas=betas)

        G_parameters = sum([list(model.parameters()) for model in self.generator], [])
        self.G_optimizer = torch.optim.Adam(G_parameters, lr=lr, betas=betas)

    def forward(self, X_a, X_b):
        z_a, mu_a = self.E_a(X_a)
        z_b, mu_b = self.E_b(X_b)

        X_ab = self.G_b(z_a)
        X_ba = self.G_a(z_b)

        return X_ab, X_ba

    def train_model(self, train_iterator, fixed_examples, n_epochs=10, print_every=500):
        output_dir = self.params['output_dir']

        global_step = 0
        for e in range(n_epochs):
            self.train()
            print('Epoch: {}\n------------------'.format(e + 1))
            for i, (X_a, X_b) in enumerate(train_iterator):
                X_a, X_b = X_a.to(self.device), X_b.to(self.device)

                # Update discriminators
                D_losses = self.update_discriminators(X_a, X_b)

                # Update generators
                G_losses = self.update_generators(X_a, X_b)

                if global_step % print_every == 0:
                    # Empty memory
                    forward_output = []
                    print('Iteration {}'.format(global_step))
                    print('Generator\n----------')
                    for k, v in G_losses.items():
                        print('{}: {:.2f}'.format(k, v))
                        self.writer.add_scalar('generator/' + k, v, global_step)
                    print('Discriminator\n-----------')
                    for k, v in D_losses.items():
                        print('{}: {:.2f}'.format(k, v))
                        self.writer.add_scalar('discriminator/' + k, v, global_step)
                    print('\n\n')

                    self.eval()
                    with torch.no_grad():
                        self.sample(fixed_examples, output_dir, e, global_step)
                    self.train()

                if global_step % 5000 == 0:
                    self.save()

                global_step += 1

            self.save()

    def forward_pass(self, X_a, X_b):
        z_a, mu_a = self.E_a(X_a)
        z_b, mu_b = self.E_b(X_b)

        X_aa = self.G_a(z_a)
        X_bb = self.G_b(z_b)

        X_ab = self.G_b(z_a)
        X_ba = self.G_a(z_b)

        z_abb, mu_abb = self.E_b(X_ab)
        z_baa, mu_baa = self.E_a(X_ba)

        X_abba = self.G_a(z_abb)
        X_baab = self.G_b(z_baa)

        return X_aa, X_bb, mu_a, mu_b, X_ab, X_ba, X_abba, X_baab, mu_abb, mu_baa

    def update_discriminators(self, X_a, X_b):
        """ Update step on a discriminator.

        Returns
        -------
            loss: float
            D(x): float
                average value of the discriminator predicted for real inputs
            D(G(x)): float
                average value of the discriminator predicted for generated inputs
        """
        self.D_optimizer.zero_grad()
        # Translation
        z_a, mu_a = self.E_a(X_a)
        z_b, mu_b = self.E_b(X_b)
        X_ab = self.G_b(z_a)
        X_ba = self.G_a(z_b)

        loss_a, D_x_a, D_G_x_a = self.params['gan_coef'] * self.D_a.discriminator_loss(X_a, X_ba.detach())
        loss_b, D_x_b, D_G_x_b = self.params['gan_coef'] * self.D_b.discriminator_loss(X_b, X_ab.detach())

        loss = loss_a + loss_b

        loss.backward()
        self.D_optimizer.step()

        losses = {'total_loss': loss.item(),
                  'loss_a': loss_a.item(),
                  'loss_b': loss_b.item(),
                  'D_x_a': D_x_a.item(),
                  'D_G_x_a': D_G_x_a.item(),
                  'D_x_b': D_x_b.item(),
                  'D_G_x_b': D_G_x_b.item()
                  }

        return losses

    # Update generators
    def update_generators(self, X_a, X_b):
        self.G_optimizer.zero_grad()

        r_coef, kl_coef, gan_coef = self.params['r_coef'], self.params['kl_coef'], self.params['gan_coef']
        X_aa, X_bb, mu_a, mu_b, X_ab, X_ba, X_abba, X_baab, mu_abb, mu_baa = self.forward_pass(X_a, X_b)

        reconst_aa = r_coef * reconst_loss(X_a, X_aa)
        reconst_bb = r_coef * reconst_loss(X_b, X_bb)
        kl_aa = kl_coef * kl_loss(mu_a)
        kl_bb = kl_coef * kl_loss(mu_b)

        reconst_abba = r_coef * reconst_loss(X_a, X_abba)
        reconst_baab = r_coef * reconst_loss(X_b, X_baab)
        kl_abba = kl_coef * kl_loss(mu_abb)
        kl_baab = kl_coef * kl_loss(mu_baa)

        gan_a = gan_coef * self.D_a.generator_loss(X_ba)
        gan_b = gan_coef * self.D_b.generator_loss(X_ab)

        loss = (reconst_aa + kl_aa
                + reconst_bb + kl_bb
                + reconst_abba + kl_abba
                + reconst_baab + kl_baab
                + gan_a + gan_b
                )

        loss.backward()
        self.G_optimizer.step()

        losses = {'total_loss': loss.item(),
                  'reconst_aa': reconst_aa.item(),
                  'reconst_bb': reconst_bb.item(),
                  'kl_aa': kl_aa.item(),
                  'kl_bb': kl_bb.item(),
                  'reconst_abba': reconst_abba.item(),
                  'reconst_baab': reconst_baab.item(),
                  'kl_abba': kl_abba.item(),
                  'kl_baab': kl_baab.item(),
                  'gan_a': gan_a,
                  'gan_b': gan_b
                  }

        return losses

    def sample(self, fixed_examples, output_dir, epoch, global_step):
        # Note that in .eval() mode, the encoder returns the mean without the noise
        examples_a, examples_b = fixed_examples
        n_examples = len(examples_a)
        dict_samples = {}
        for key in ['X_a', 'X_aa', 'X_ab', 'X_abba', 'X_b', 'X_bb', 'X_ba', 'X_baab']:
            dict_samples[key] = []

        examples_a, examples_b = examples_a.to(self.device), examples_b.to(self.device)

        for i in range(n_examples):
            X_a, X_b = examples_a[i].unsqueeze(0), examples_b[i].unsqueeze(0)
            X_aa, X_bb, mu_a, mu_b, X_ab, X_ba, X_abba, X_baab, mu_abb, mu_baa = self.forward_pass(X_a, X_b)

            dict_samples['X_a'].append(X_a)
            dict_samples['X_aa'].append(X_aa)
            dict_samples['X_ab'].append(X_ab)
            dict_samples['X_abba'].append(X_abba)

            dict_samples['X_b'].append(X_b)
            dict_samples['X_bb'].append(X_bb)
            dict_samples['X_ba'].append(X_ba)
            dict_samples['X_baab'].append(X_baab)

        filename_suffix = 'epoch_{:02d}_iter_{:06d}.jpg'.format(epoch, global_step)
        # Save A to B images
        images_a_to_b = []
        for i in range(n_examples):
            images_a_to_b.append(dict_samples['X_a'][i])
            images_a_to_b.append(dict_samples['X_aa'][i])
            images_a_to_b.append(dict_samples['X_ab'][i])
            images_a_to_b.append(dict_samples['X_abba'][i])

        images_a_to_b = torch.cat(images_a_to_b)

        grid_a_to_b = vutils.make_grid(images_a_to_b, nrow=4, padding=0, normalize=True)
        vutils.save_image(grid_a_to_b, os.path.join(output_dir, 'a_to_b_' + filename_suffix))

        # Save B to A images
        images_b_to_a = []
        for i in range(n_examples):
            images_b_to_a.append(dict_samples['X_b'][i])
            images_b_to_a.append(dict_samples['X_bb'][i])
            images_b_to_a.append(dict_samples['X_ba'][i])
            images_b_to_a.append(dict_samples['X_baab'][i])

        images_b_to_a = torch.cat(images_b_to_a)
        grid_b_to_a = vutils.make_grid(images_b_to_a, nrow=4, padding=0, normalize=True)
        vutils.save_image(grid_b_to_a, os.path.join(output_dir, 'b_to_a_' + filename_suffix))

    def save(self):
        checkpoint_name = os.path.join(self.params['output_dir'], 'model.pt')
        torch.save({'E_a': self.E_a.state_dict(),
                    'E_b': self.E_b.state_dict(),
                    'G_a': self.G_a.state_dict(),
                    'G_b': self.G_b.state_dict(),
                    'D_a': self.D_a.state_dict(),
                    'D_b': self.D_b.state_dict(),
                    'G_optimizer': self.G_optimizer.state_dict(),
                    'D_optimizer': self.D_optimizer.state_dict()},
                   checkpoint_name)

    def load(self, checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        self.E_a.load_state_dict(checkpoint['E_a'])
        self.E_b.load_state_dict(checkpoint['E_b'])
        self.G_a.load_state_dict(checkpoint['G_a'])
        self.G_b.load_state_dict(checkpoint['G_b'])
        self.D_a.load_state_dict(checkpoint['D_a'])
        self.D_b.load_state_dict(checkpoint['D_b'])

        self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])


class VAE(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.encoder = Encoder(device)
        self.decoder = Decoder()

    def forward(self, x):
        z, mean = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mean, z



