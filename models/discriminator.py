# Code based on https://github.com/tensorflow/models/blob/master/research/slim/nets/dcgan.py
from math import log
import tensorflow as tf

slim = tf.contrib.slim


class Discriminator(object):
    def __init__(self, args, image_shape):
        self.var_list = None

        self.fake = tf.placeholder(
                tf.float32, [None] + list(image_shape), name='c')
        self.real = tf.placeholder(
                tf.float32, [None] + list(image_shape), name='c')
        self.z = tf.placeholder(tf.float32, [
                args.discrim_batch_size, args.z_dim], name='z')

        self.real_probs, self.real_logits = self.build_model(self.real)
        self.fake_probs, self.fake_logits = self.build_model(self.fake, reuse=True)

        self.d_loss_real = - tf.reduce_mean(self.real_logits)
        self.d_loss_fake = tf.reduce_mean(self.fake_logits)

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = -self.d_loss_fake

        self.build_optim()

    def build_optim(self):
        pass

    def build_model(self,
                    inputs,
                    depth=64,
                    is_training=True,
                    reuse=False,
                    scope='Discriminator',
                    data_format='channels_first',
                    fused_batch_norm=False):
        normalizer_fn = slim.batch_norm
        normalizer_fn_args = {
                'is_training': is_training,
                'zero_debias_moving_mean': True,
                'fused': fused_batch_norm,
        }

        inp_shape = inputs.get_shape().as_list()[1]

        with tf.variable_scope(scope, values=[inputs], reuse=reuse) as scope:
            with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
                with slim.arg_scope([slim.conv2d],
                                    stride=2,
                                    kernel_size=4,
                                    activation_fn=tf.nn.leaky_relu):
                    net = inputs
                    for i in range(int(log(inp_shape, 2))):
                        scope = f'conv{i+1}'
                        current_depth = depth * 2**i
                        normalizer_fn_ = None if i == 0 else normalizer_fn
                        net = slim.conv2d(
                                net, current_depth,
                                normalizer_fn=normalizer_fn_, scope=scope)

                    logits = slim.conv2d(
                            net, 1, kernel_size=1, stride=1, padding='VALID',
                            normalizer_fn=None, activation_fn=None)

                    logits = tf.reshape(logits, [-1, 1])
                    probs = tf.nn.sigmoid(logits)

            if not reuse:
                assert self.var_list is None, "self.var_list should be set only once"
                self.var_list = tf.trainable_variables(scope=None)

        return probs, logits

