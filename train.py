import os
import argparse
import torch

from model import UNIT
from dataset import get_data

ROOT = '/data/cvfs/ah2029/datasets/bdd100k/'


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--gpu', type=str, required=True)
parser.add_argument('--default_init', action='store_true')
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--resume', action='store_true')

args = parser.parse_args()
output_dir = args.output_dir
gpu = args.gpu
n_epochs = args.n_epochs
batch_size = args.batch_size
print_every = args.print_every

train_iterator, val_iterator, fixed_examples = get_data(ROOT, batch_size=batch_size,
                                                        img_size=(512, 512), subset=1)

print('Number of samples: {}'.format(len(train_iterator)))

params = {'output_dir': output_dir,
          'r_coef': 10,
          'kl_coef': 0.01,
          'gan_coef': 1,
          'betas': (0.5, 0.999),
          'lr': 1e-4,
          'device': torch.device('cuda:' + gpu),
          'default_init': args.default_init
          }

unit_network = UNIT(params)

if args.resume:
    checkpoint_name = os.path.join(params['output_dir'], 'model.pt')
    print('Loading model from {}'.format(checkpoint_name))
    unit_network.load(checkpoint_name)

unit_network.train_model(train_iterator, fixed_examples, n_epochs, print_every=print_every)