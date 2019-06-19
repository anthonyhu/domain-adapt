import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def preprocess_image(image, out_size):
    """ Tensorflow image preprocessing: bilinear resizing, and normalising to [-1, 1]

    Parameters
    ----------
        image: tf.Tensor
            output from tf.read_file

        outsize: tuple(int, int)
             defined as (height, width)

    Returns
        resized_image: tf.Tensor
    """
    # Original image is (720, 1280)
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize shortest side to 512
    new_size = (512, 910)
    image = tf.image.resize_images(image, new_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.image.random_crop(image, (*out_size, 3))
    image = tf.image.random_flip_left_right(image)

    image = (2 * image - 255.0) / 255.0  # normalise to [-1, 1] range

    return image


def load_and_preprocess_image(path, out_size):
    image = tf.read_file(path)
    return preprocess_image(image, out_size)


def preprocess_vgg(x):
    x = 255 * (x + 1) / 2  # [-1,1] to [0, 255]
    x = x[..., ::-1]  # RGB to BGR
    # Substract mean
    mean = K.constant([103.939, 116.779, 123.680])
    mean = K.reshape(mean, (1, 1, 1, 3))
    return x - mean


def convert_to_uint8(img):
    """ Convert image from floating point [-1, 1] to np.uint8 [0, 255]"""
    return np.uint8(255 * (img + 1) / 2)


def write_log(callback, names, logs, batch_number):
    """ Write logs to tensorboard

    Parameters
    ----------
        callback: tf.keras.callbacks.TensorBoard
        names: list<str>
            names of the scalars to save
        logs: list<float>
            values of the scalars to save
        batch_number: int
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_number)
        callback.writer.flush()
