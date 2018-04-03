# Code based on https://github.com/tensorflow/models/blob/master/research/slim/nets/dcgan.py
from math import log
import tensorflow as tf

slim = tf.contrib.slim


class Discriminator(object):
    def __init__(self, args, image_shape, scope_name):
        self.args = args
        self.var_list = None
        self.scope_name = scope_name

        self.fake = tf.placeholder(
                tf.float32, [None] + list(image_shape), name='c')
        self.real = tf.placeholder(
                tf.float32, [None] + list(image_shape), name='c')

        self.real_probs, self.real_logits = self.build_model(self.real)
        self.fake_probs, self.fake_logits = self.build_model(self.fake, reuse=True)

        self.build_optim()

    def build_model(self,
                    inputs,
                    depth=64,
                    is_training=True,
                    reuse=False,
                    data_format='channels_first',
                    fused_batch_norm=False):
        normalizer_fn = slim.batch_norm
        normalizer_fn_args = {
                'is_training': is_training,
                'zero_debias_moving_mean': True,
                'fused': fused_batch_norm,
        }

        inp_shape = inputs.get_shape().as_list()[1]

        with tf.variable_scope(self.scope_name, values=[inputs], reuse=reuse) as scope:
            self.step = tf.get_variable(
                    "disc_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)

            with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
                with slim.arg_scope([slim.conv2d],
                                    stride=2,
                                    kernel_size=4,
                                    activation_fn=tf.nn.leaky_relu):
                    net = inputs
                    for i in range(int(log(inp_shape, 2))):
                        scope = 'conv{}'.format(i+1)
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
                self.var_list = tf.trainable_variables(self.scope_name)

        return probs, logits


    def build_optim(self):
        ##################
        # Optimizers
        ##################

        self.d_loss_real = - tf.reduce_mean(self.real_logits)
        self.d_loss_fake = tf.reduce_mean(self.fake_logits)

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = - self.d_loss_fake

        epsilon = tf.random_uniform([], 0.0, 1.0)
        interpolates = epsilon * self.real + (1 - epsilon) * self.fake
        d_hat = self.build_model(interpolates, reuse=True)

        gradients = tf.gradients(d_hat, interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0) * 10)

        self.d_loss += gradient_penalty

        self.opt = tf.train.AdamOptimizer(
                self.args.disc_lr, name="policy_optim")
        self.train_op = self.opt.minimize(
                self.d_loss, self.step, var_list=self.var_list)

        if self.args.clip_disc_weights:
            clip_ops = []
            for var in self.var_list:
                clip_bounds = [-.01, .01]
                clip_ops.append(
                    tf.assign(
                        var, 
                        tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                    )
                )
            self.clip_disc_weights = tf.group(*clip_ops)
        else:
            self.clip_disc_weights = None

        ##################
        # Summaries
        ##################

        self.summary_op = tf.summary.merge([
                tf.summary.scalar("gan/disc_loss", self.d_loss),
                tf.summary.scalar("gan/gen_loss", self.g_loss),
        ])

