import os
import json
import random

import numpy as np
import tensorflow as tf
import torchvision.transforms as transforms

from .utils import load_and_preprocess_image
from PIL import Image

root = '/data/cvfs/ah2029/datasets/bdd100k/'


def load_day_and_night(split='train', subset=1.0):
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


def create_dataset(X_day, X_night, image_size=(512, 512), batch_size=1):
    """ Create a tensorflow dataset with a list of filenames and image size

    Parameters
    ----------
        X_day: list<str>
        X_night: list<str>
        image_size: tuple(int, int)
            defined as height, width

    Returns
    -------
        dataset: tf.data.Dataset

    """
    def map_(x, y, out_size):
        img_day = load_and_preprocess_image(x, out_size)
        img_night = load_and_preprocess_image(y, out_size)
        return (img_day, img_night), (img_day, img_night)

    dataset = tf.data.Dataset.from_tensor_slices((X_day, X_night))
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(X_day)))
    dataset = dataset.map(lambda x, y: map_(x, y, image_size))
    dataset = dataset.batch(batch_size)
    return dataset


def load_batch(filenames, image_size):
    """ Load a batch of images

    Parameters
    ----------
        filenames: list<str>
        image_size: tuple(int, int)
    """
    assert image_size[0] == image_size[1]
    data_transforms = transforms.Compose([transforms.Resize(image_size[0]),
                                          transforms.RandomCrop(image_size[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch = []
    for filename in filenames:
        img = Image.open(filename)
        img = data_transforms(img)
        # Channel last
        img = np.transpose(img, (1, 2, 0))
        batch.append(img)
    return np.stack(batch, axis=0)