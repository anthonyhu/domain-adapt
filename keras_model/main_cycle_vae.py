import os
import argparse
import tensorflow as tf

from keras_model.model import CycleVAE

EXPERIMENT_ROOT = '/data/cvfs/ah2029/experiments/bdd100k/'
EPOCHS = 10
IMAGE_SIZE = (512, 512)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='directory of the config file')
    parser.add_argument('--gpu', type=str, required=True, help='gpu to use')
    parser.add_argument('--batch', type=int, required=False, default=1, help='gpu to use')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_folder = os.path.join(EXPERIMENT_ROOT, args.dir)
    os.makedirs(save_folder, exist_ok=True)
    batch_size = args.batch

    tf.logging.set_verbosity(tf.logging.ERROR)  # to get rid of the warnings
    model = CycleVAE(save_folder=save_folder, image_size=IMAGE_SIZE)
    model.train(epochs=EPOCHS, batch_size=batch_size)
