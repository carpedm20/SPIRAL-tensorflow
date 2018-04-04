# Code based on https://github.com/tensorflow/models/blob/master/research/slim/nets/dcgan.py
from math import log
import tensorflow as tf

import utils as ut

tl = tf.layers


class Discriminator(object):
    def __init__(self, args, step, image_shape, norm_fn, scope_name):
        self.args = args
        self.step = step
        self.scope_name = scope_name

        self.fake = fake = tf.placeholder(
                tf.float32, [None] + list(image_shape), name='c')
        self.real = real = tf.placeholder(
                tf.float32, [None] + list(image_shape), name='c')

        # NHWC -> NCHW
        if args.num_gpu > 0:
            fake = tf.transpose(fake, [0, 3, 1, 2])
            real = tf.transpose(real, [0, 3, 1, 2])
            self.data_format = "channels_first"
        else:
            self.data_format = "channels_last" * 10

        if norm_fn is not None:
            fake = norm_fn(fake)
            real = norm_fn(real)

        self.fake_in = fake
        self.real_in = real

        self.real_probs, self.real_logits = self.build_model(self.real_in)
        self.fake_probs, self.fake_logits = self.build_model(self.fake_in, reuse=True)

        self.build_optim()

    def predict(self, images):
        sess = tf.get_default_session()

        feed_dict = {
                self.real: images,
        }
        probs = sess.run(self.real_probs, feed_dict)

        return probs

    def build_model(self,
                    inputs,
                    depth=32,
                    is_training=True,
                    reuse=False):

        inp_shape = inputs.get_shape().as_list()[2]

        with tf.variable_scope(self.scope_name, values=[inputs], reuse=reuse) as scope:
            x = inputs

            layer_num = int(log(inp_shape, 2))
            for idx in range(layer_num):
                cur_depth = depth * 2**idx

                x = tl.conv2d(
                        x, cur_depth, 4,
                        strides=(2, 2),
                        padding='same',
                        activation=None,
                        data_format=self.data_format,
                        name="conv{}".format(idx))

                if idx > 0:
                    x = tl.batch_normalization(
                            x, axis=1 if self.data_format == "channels_first" else -1,
                            fused=True, training=True)

                x = tf.nn.leaky_relu(x)

            logits = tl.conv2d(
                    x, 1, 1,
                    strides=(1, 1),
                    padding='valid',
                    activation=None,
                    data_format=self.data_format,
                    name="conv{}".format(layer_num))

            logits = tf.reshape(logits, [-1, 1])
            probs = tf.nn.sigmoid(logits)

        return probs, logits

    def build_optim(self):
        self.g_loss = - tf.reduce_mean(self.fake_logits)
        self.critic_loss = tf.reduce_mean(self.fake_logits) - tf.reduce_mean(self.real_logits)

        real_batch, fake_batch = tf.shape(self.real_in)[0], tf.shape(self.fake_in)[0]
        fake_repeat = int(self.args.real_batch_size / self.args.fake_batch_size) 

        fake_in = tf.concat(
                [self.fake_in] * fake_repeat + \
                [self.fake_in[:real_batch - fake_repeat * fake_batch]], 0)

        alpha = tf.random_uniform([self.args.real_batch_size, 1, 1, 1], 0.0, 1.0)
        interpolates = alpha * self.real_in + (1 - alpha) * fake_in

        gradients = tf.gradients(
                self.build_model(interpolates, reuse=True)[1], interpolates)[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.)) * 20

        self.d_loss = self.critic_loss + self.gradient_penalty

        self.var_list = tf.trainable_variables(self.scope_name)

        self.opt = tf.train.AdamOptimizer(
                self.args.disc_lr, beta1=0.5, beta2=0.9, name="disc_optim")

        update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope=self.scope_name)

        with tf.control_dependencies(update_ops):
            self.train_op = self.opt.minimize(
                    self.d_loss, self.step, var_list=self.var_list)

        self.var_list += self.opt.variables()

        scope_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
        self.var_list += [
            v for v in scope_vars if "batch_normalization" in v.name]

        ##################
        # Summaries
        ##################

        if self.scope_name == 'local':
            self.summary_op = tf.summary.merge([
                    tf.summary.scalar("gan/critic_loss", self.critic_loss),
                    tf.summary.scalar("gan/disc_loss", self.d_loss),
                    tf.summary.scalar("gan/penalty", self.gradient_penalty),
                    tf.summary.scalar("gan/gen_loss", self.g_loss),
            ])

