import tensorflow as tf
tfgan = tf.contrib.gan


def get_image_grid(images, batch_size, num_classes, num_images_per_class):
    # Code from https://github.com/tensorflow/models/blob/master/research/gan/cifar/util.py
    images.shape[0:1].assert_is_compatible_with([batch_size])
    if batch_size < num_classes * num_images_per_class:
        raise ValueError('Not enough images in batch to show the desired number of '
                                         'images.')
    if batch_size % num_classes != 0:
        raise ValueError('`batch_size` must be divisible by `num_classes`.')

    # Only get a certain number of images per class.
    num_batches = batch_size // num_classes
    indices = [i * num_batches + j for i in xrange(num_classes)
                         for j in xrange(num_images_per_class)]
    sampled_images = tf.gather(images, indices)
    return tfgan.eval.image_reshaper(
            sampled_images, num_cols=num_images_per_class)

