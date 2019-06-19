import tensorflow.keras.backend as K


def reconst_loss(inputs, outputs):
    """
    Parameters
    ----------
        inputs: tf.Tensor (-1, h, w, 3)
        outputs: tf.Tensor (-1, h, w, 3)

    Returns
    -------
        reconst_loss: tf.Tensor
            L1 reconstruction loss
    """
    r_loss = K.abs(inputs - outputs)
    return K.mean(r_loss)


def kl_div_loss(z_mean):
    """
    Parameters
    ----------
        z_mean: tf.Tensor (-1, h, w, d)

    Returns
    -------
        kl_loss: tf.Tensor
            KL divergence loss
    """
    kl_loss = 0.5 * K.square(z_mean)
    return K.mean(kl_loss)


def compute_vgg_loss(features1, features2):
    return K.mean(K.square(features1 - features2))
