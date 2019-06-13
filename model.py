import os
import torch

import torch.nn as nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from layers import Encoder, Decoder, Discriminator
from losses import reconst_loss, kl_loss
from utils import convert_to_pil


class UNIT():
    def __init__(self, params):
        self.params = params

        # Tensorboard
        output_dir = params['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self.writer = SummaryWriter(output_dir, flush_secs=60)

        # Create UNIT network
        self.device = torch.device('cuda')
        self.E_a = Encoder(self.device).to(self.device)
        self.E_b = Encoder(self.device).to(self.device)

        self.G_a = Decoder().to(self.device)
        self.G_b = Decoder().to(self.device)

        self.D_a = Discriminator().to(self.device)
        self.D_b = Discriminator().to(self.device)

        self.generator = [self.E_a, self.E_b, self.G_a, self.G_b]
        self.discriminator = [self.D_a, self.D_b]

        # Optimizers
        betas = params['betas']
        lr = params['lr']
        D_parameters = sum([list(model.parameters()) for model in self.discriminator], [])
        self.D_optimizer = torch.optim.Adam(D_parameters, lr=lr, betas=betas)

        G_parameters = sum([list(model.parameters()) for model in self.generator], [])
        self.G_optimizer = torch.optim.Adam(G_parameters, lr=lr, betas=betas)

    def train_model(self, train_iterator, n_epochs=10, print_every=500):
        output_dir = self.params['output_dir']
        fixed_X_a, fixed_X_b = next(iter(train_iterator))
        fixed_X_a, fixed_X_b = fixed_X_a.to(self.device), fixed_X_b.to(self.device)

        global_step = 0
        for e in range(n_epochs):
            self.set_train_mode()
            print('Epoch: {}\n------------------'.format(e + 1))
            for i, (X_a, X_b) in enumerate(train_iterator):
                X_a, X_b = X_a.to(self.device), X_b.to(self.device)

                forward_output = self.forward_pass(X_a, X_b)

                D_losses = self.update_discriminators(X_a, X_b, forward_output)

                # Update generators
                G_losses = self.update_generators(X_a, X_b, forward_output)

                if i % print_every == 0:
                    print('Iteration {}'.format(i))
                    print('Generator\n----------')
                    for k, v in G_losses.items():
                        print('{}: {:.2f}'.format(k, v))
                        self.writer .add_scalar('generator/' + k, v, global_step)
                    print('Discriminator\n-----------')
                    for k, v in D_losses.items():
                        print('{}: {:.2f}'.format(k, v))
                        self.writer.add_scalar('discriminator/' + k, v, global_step)

                    self.set_eval_mode()
                    self.sample(fixed_X_a, fixed_X_b, output_dir, e, global_step)
                    self.set_train_mode()

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

    def update_discriminators(self, X_a, X_b, forward_output):
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
        X_aa, X_bb, mu_a, mu_b, X_ab, X_ba, X_abba, X_baab, mu_abb, mu_baa = forward_output

        # A model
        D_real_a = self.D_a(X_a)
        D_fake_a = self.D_a(X_ba.detach())

        valid = torch.ones_like(D_real_a).to(self.device)
        fake = torch.zeros_like(D_fake_a).to(self.device)

        loss_real_a = torch.nn.MSELoss()(D_real_a, valid)
        loss_fake_a = torch.nn.MSELoss()(D_fake_a, fake)
        loss_a = self.params['gan_coef'] * (loss_real_a + loss_fake_a) / 2

        # B model
        D_real_b = self.D_b(X_b)
        D_fake_b = self.D_b(X_ab.detach())

        loss_real_b = torch.nn.MSELoss()(D_real_b, valid)
        loss_fake_b = torch.nn.MSELoss()(D_fake_b, fake)
        loss_b = self.params['gan_coef'] * (loss_real_b + loss_fake_b) / 2

        loss = loss_a + loss_b

        loss.backward()
        self.D_optimizer.step()

        losses = {'loss_a': loss_a.item(),
                  'loss_b': loss_b.item(),
                  'D_x_a': D_real_a.mean().item(),
                  'D_G_x_a': D_fake_a.mean().item(),
                  'D_x_b': D_real_b.mean().item(),
                  'D_G_x_b': D_fake_b.mean().item()
                  }

        return losses

    # Update generators
    def update_generators(self, X_a, X_b, forward_output):
        self.G_optimizer.zero_grad()

        r_coef, kl_coef, gan_coef = self.params['r_coef'], self.params['kl_coef'], self.params['gan_coef']
        X_aa, X_bb, mu_a, mu_b, X_ab, X_ba, X_abba, X_baab, mu_abb, mu_baa = forward_output

        reconst_aa = r_coef * reconst_loss(X_a, X_aa)
        reconst_bb = r_coef * reconst_loss(X_b, X_bb)
        kl_aa = kl_coef * kl_loss(mu_a)
        kl_bb = kl_coef * kl_loss(mu_b)

        reconst_abba = r_coef * reconst_loss(X_a, X_abba)
        reconst_baab = r_coef * reconst_loss(X_b, X_baab)
        kl_abba = kl_coef * kl_loss(mu_abb)
        kl_baab = kl_coef * kl_loss(mu_baa)

        D_gen_a = self.D_a(X_ba)
        D_gen_b = self.D_b(X_ab)
        valid = torch.ones_like(D_gen_a).to(self.device)

        gan_a = gan_coef * torch.nn.MSELoss()(D_gen_a, valid)
        gan_b = gan_coef * torch.nn.MSELoss()(D_gen_b, valid)

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

    def sample(self, X_a, X_b, output_dir, epoch, global_step):
        X_aa, X_bb, mu_a, mu_b, X_ab, X_ba, X_abba, X_baab, mu_abb, mu_baa = self.forward_pass(X_a, X_b)

        plt.figure(figsize=(10, 5))
        plt.subplot(231)
        plt.imshow(convert_to_pil(X_a[0].cpu()))
        plt.subplot(232)
        plt.imshow(convert_to_pil(X_ab[0].detach().cpu()))
        plt.subplot(233)
        plt.imshow(convert_to_pil(X_abba[0].detach().cpu()))

        plt.subplot(234)
        plt.imshow(convert_to_pil(X_b[0].cpu()))
        plt.subplot(235)
        plt.imshow(convert_to_pil(X_ba[0].detach().cpu()))
        plt.subplot(236)
        plt.imshow(convert_to_pil(X_baab[0].detach().cpu()))
        filename = os.path.join(output_dir, 'epoch_{:02d}_iter_{:06d}.png'.format(epoch, global_step))
        plt.savefig(filename)
        plt.show()
        plt.close()

    def set_train_mode(self):
        for model in (self.generator + self.discriminator):
            model.train()

    def set_eval_mode(self):
        for model in (self.generator + self.discriminator):
            model.eval()

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



