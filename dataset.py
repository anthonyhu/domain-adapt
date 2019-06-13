import os
import json
import random

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


def load_day_and_night(root, split='train', subset=1.0):
    """ Load image filenames with clear or partly cloudy weather, and separate in day and night. BDD100k dataset.

    Parameters
    ----------
        split: str

    Returns
    -------
        X_day: np.array<str>
            filenames corresponding to the images in daytime
        X_night: np.array<str>
            filenames corresponding to the images at night
    """
    with open(os.path.join(root, 'labels/bdd100k_labels_images_' + split + '.json')) as json_file:
        bdd_labels = json.load(json_file)

    X_day = []
    X_night = []
    for label in bdd_labels:
        weather = label['attributes']['weather']
        timeofday = label['attributes']['timeofday']
        filename = os.path.join(root, 'images/100k/', split, label['name'])
        if weather in ['clear', 'partly cloudy']:
            if timeofday == 'daytime':
                X_day.append(filename)
            elif timeofday == 'night':
                X_night.append(filename)

    if subset < 1.0:
        n_day = int(subset * len(X_day))
        n_night = int(subset * len(X_night))
        X_day = random.sample(X_day, n_day)
        X_night = random.sample(X_night, n_night)

    # Make the two lists have the same size
    n = min(len(X_day), len(X_night))
    X_day = X_day[:n]
    X_night = X_night[:n]

    return np.array(X_day), np.array(X_night)


class DomainDataset(Dataset):
    def __init__(self, X_day, X_night, transforms=None):
        super(DomainDataset, self).__init__()
        assert len(X_day) == len(X_night)
        self.X_day = X_day
        self.X_night = X_night
        self.transforms = transforms

    def __len__(self):
        return len(self.X_day)

    def __getitem__(self, idx):
        x_day = Image.open(self.X_day[idx])
        x_night = Image.open(self.X_night[idx])
        if self.transforms:
            x_day = self.transforms(x_day)
            x_night = self.transforms(x_night)

        return x_day, x_night


def get_data(root, batch_size=1, img_size=(512, 512)):
    """ Create train/val iterator.
        Transformations are: CenterCrop, Resize, Convert to tensor in range [0, 1], Normalize to [-1, 1]
    """
    X_day_train, X_night_train = load_day_and_night(root, 'train', subset=0.1)
    X_day_val, X_night_val = load_day_and_night(root, 'val', subset=0.1)

    # Original image is (1280, 720)
    data_transforms = transforms.Compose([transforms.CenterCrop(720),
                                          transforms.Resize(img_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = DomainDataset(X_day_train, X_night_train, data_transforms)
    val_dataset = DomainDataset(X_day_val, X_night_val, data_transforms)

    train_iterator = DataLoader(train_dataset, batch_size, shuffle=True)
    val_iterator = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_iterator, val_iterator