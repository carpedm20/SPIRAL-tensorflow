# Code based on https://github.com/tensorflow/models/blob/master/research/slim/nets/dcgan.py
from math import log
import tensorflow as tf

import utils as ut

tl = tf.layers
logger = ut.logging.get_logger()


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
            channel_idx = 1
        else:
            self.data_format = "channels_last"
            channel_idx = -1

        if norm_fn is not None:
            fake = norm_fn(fake)
            real = norm_fn(real)

        if self.args.conditional:
            fake = tf.concat([fake, real], axis=channel_idx)
            real = tf.concat([real, real], axis=channel_idx)

        self.fake_in = fake
        self.real_in = real


        self.real_probs, self.real_logits = self.build_model(self.real_in)
        self.var_list = tf.trainable_variables(self.scope_name)

        self.fake_probs, self.fake_logits = self.build_model(self.fake_in, reuse=True)

        self.build_optim()

    def build_model(self,
                    inputs,
                    is_training=True,
                    reuse=False):

        inp_shape = inputs.get_shape().as_list()[2]

        with tf.variable_scope(self.scope_name, values=[inputs], reuse=reuse) as scope:
            x = inputs

            layer_num = int(log(inp_shape, 2))
            for idx in range(layer_num):
                cur_depth = self.args.disc_dim * 2**idx

                x = tl.conv2d(
                        x, cur_depth, 5,
                        strides=(2, 2),
                        padding='same',
                        activation=None,
                        data_format=self.data_format,
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        name="conv{}".format(idx))

                logger.info("conv: {} ({})".format(x.name, x.get_shape()))

                if idx > 0 and self.args.disc_batch_norm:
                    x = tl.batch_normalization(
                            x, axis=1 if self.data_format == "channels_first" else -1,
                            fused=True, training=True)

                x = tf.nn.leaky_relu(x)

            x = tl.flatten(x)
            logits = tl.dense(
                    x, 1,
                    activation=None,
                    kernel_initializer=tf.keras.initializers.glorot_normal(),
                    name="dense")

            logger.info("logits: {} ({})".format(logits.name, logits.get_shape()))

            logits = tf.reshape(logits, [-1])
            probs = tf.nn.sigmoid(logits)

        return probs, logits

    def build_optim(self):
        self.g_loss = -tf.reduce_mean(self.fake_logits)
        self.critic_loss = \
                tf.reduce_mean(self.fake_logits) - tf.reduce_mean(self.real_logits)

        alpha = tf.random_uniform(
                [self.args.disc_batch_size, 1],
                minval=0.0,
                maxval=1.0)

        fake_data = tl.flatten(self.fake_in)
        real_data = tl.flatten(self.real_in)

        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)

        diff_in = tf.reshape(interpolates, ut.tf.int_shape(self.fake_in))
        diff_probs, diff_logits = self.build_model(diff_in, reuse=True)

        gradients = tf.gradients(diff_probs, [interpolates])[0]
        slopes = tf.sqrt(1e-8+tf.reduce_sum(
                tf.square(gradients), reduction_indices=[1]))

        self.gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        self.d_loss = self.critic_loss + \
                self.args.wgan_lambda * self.gradient_penalty

        self.opt = tf.train.AdamOptimizer(
                self.args.disc_lr, beta1=0.5, beta2=0.9,
                name="disc_optim")

        if self.args.disc_batch_norm:
            update_ops = tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS, scope=self.scope_name)
            with tf.control_dependencies(update_ops):
                self.train_op = self.opt.minimize(
                        self.d_loss, self.step, var_list=self.var_list)
        else:
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

    def predict(self, images):
        sess = tf.get_default_session()
        feed_dict = {
                self.real: images,
        }
        probs = sess.run(self.real_probs, feed_dict)
        return probs


if __name__ == '__main__':
    from config import get_args
    args = get_args()

    noise = tf.random_normal([args.disc_batch_size, 128])

    x = tl.dense(
            noise, 4*4*4*2048,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            name="dense")

    x
