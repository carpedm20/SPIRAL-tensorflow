import numpy as np
import tensorflow as tf

import utils as ut

tl = tf.layers
tc = tf.nn.rnn_cell


class Policy(object):

    def __init__(self, args, env, scope_name,
                 image_shape, action_sizes, data_format):
        self.args = args
        scale = args.scale
        self.lstm_size = args.lstm_size

        if data_format == 'channels_first' and args.dynamic_channel:
            self.image_shape = list(image_shape[-1:] + image_shape[:-1])
        else:
            self.image_shape = list(image_shape)

        self.action_sizes = action_sizes
        self.data_format = data_format

        action_num = len(action_sizes)

        with tf.variable_scope(scope_name) as scope:
            # [B, max_time, H, W, C]
            self.x = x = tf.placeholder(
                    tf.float32, [None, None] + self.image_shape, name='x')

            # last is only used for summary
            x = x[:,:env.episode_length]

            # Flatten multiple episodes
            # XXX: important difference from openai/universe-starter-agent
            x_shape = tf.shape(x)
            batch_size, max_time = x_shape[0], x_shape[1]

            # [B, max_time, action_num]
            self.ac = ac = tf.placeholder(
                    tf.float32, [None, None, action_num], name='ac')

            if args.conditional:
                # [B, max_time, H, W, C]
                self.c = c = tf.placeholder(
                        tf.float32, [None, None] + self.image_shape, name='c')
                # TODO: need to get confirmed from the authors
                x = tf.concat([x, c], axis=-1)
                x_shape = list(self.image_shape)
                x_shape[-1] = int(x.get_shape()[-1])

                self.z = None
            else:
                self.c = None
                x_shape = self.image_shape

                # [B, max_time, z_dim]
                self.z = z = tf.placeholder(
                        tf.float32, [None, None, self.args.z_dim], name='z')

                z_enc = mlp(
                        z,
                        self.lstm_size,
                        name="z_enc")

            x = tf.reshape(x, [-1] + x_shape)
            ac = tf.reshape(ac, [-1, action_num])

            if data_format == 'channels_first' and args.dynamic_channel:
                x = tf.transpose(x, [0, 3, 1, 2])

            ################################
            # Beginning of policy network
            ################################

            a_enc = mlp(
                    tf.expand_dims(ac, -1),
                    int(16),
                    name="a_enc")
            a_concat = tf.reshape(
                    a_enc, [-1, int(16) * action_num])
            a_fc = tl.dense(
                    a_concat, int(32),
                    activation=tf.nn.relu,
                    name="a_concat_fc")

            # [B, 1, 1, 32]
            a_expand = tf.expand_dims(tf.expand_dims(a_fc, 1), 1)
            if data_format == 'channels_first' and args.dynamic_channel:
                a_expand = tf.transpose(a_expand, [0, 3, 1, 2])

            x_enc = tl.conv2d(
                    x, int(32), 5,
                    padding='same',
                    activation=tf.nn.relu,
                    data_format=self.data_format,
                    name="x_c_enc" if args.conditional else "x_enc")

            add = x_enc + a_expand

            for idx in range(int(3)):
                add = tl.conv2d(
                        add, int(32), 4, strides=(2, 2),
                        padding='valid',
                        activation=tf.nn.relu,
                        data_format=self.data_format,
                        name="add_enc_{}".format(idx))

            for idx in range(int(8*scale)):
                add = res_block(
                        add, 32, 3, self.data_format,
                        name="encoder_res_{}".format(idx))

            flat = tl.flatten(add)

            out = tl.dense(
                    flat, self.lstm_size,
                    activation=tf.nn.relu,
                    name="flat_fc")

            # [batch_size, max_time, ...]
            flat_out = tl.flatten(out)
            lstm_in_shape = [batch_size, max_time, flat_out.get_shape()[-1]]
            lstm_in = tf.reshape(flat_out, lstm_in_shape, name="lstm_in")

            if not self.args.conditional:
                lstm_in += z_enc
            
            self.lstm = tc.BasicLSTMCell(self.lstm_size, state_is_tuple=True)

            def make_init(batch_size):
                c_init = np.zeros((batch_size, self.lstm.state_size.c), np.float32)
                h_init = np.zeros((batch_size, self.lstm.state_size.h), np.float32)
                return [c_init, h_init]

            self.state_init = ut.misc.keydefaultdict(make_init)

            c_in = tf.placeholder(
                    tf.float32,
                    [None, self.lstm.state_size.c],
                    name="lstm_c_in")
            h_in = tf.placeholder(
                    tf.float32,
                    [None, self.lstm.state_size.h],
                    name="lstm_h_in")
            self.state_in = [c_in, h_in]
            state_in = tc.LSTMStateTuple(c_in, h_in)

            lstm_out, lstm_state = tf.nn.dynamic_rnn(
                    self.lstm,
                    # [batch_size, max_time, ...]
                    lstm_in,
                    # [batch_size, cell.state_size]
                    initial_state=state_in,
                    time_major=False)

            # [bach_size, max_time, action_size]
            self.one_hot_samples, self.samples, self.logits = self.decoder(
                    tf.nn.relu(lstm_out), self.action_sizes,
                    self.data_format, self.lstm_size, scale)

            lstm_c, lstm_h = lstm_state
            self.state_out = [lstm_c, lstm_h]

            self.vf = tl.dense(
                    lstm_out, 1,
                    activation=None,
                    name="value")[:,:,0]
                    #kernel_initializer=normalized_columns_initializer(1.0))[:,:,0]

            self.var_list = tf.trainable_variables(scope=scope_name)

    def get_initial_features(self, batch_size, flat=False):
        assert batch_size == 1 and flat, \
                "Use flat=True only when batch_size == 1"
        out = self.state_init[batch_size]
        if flat:
            out = [out[0][0], out[1][0]]
        return out

    def get_feed_dict(self, ob, ac, c, h, condition, z):
        feed_dict = {
                self.x: [[ob]], # fake batch, time axis
                self.ac: [[ac]],
                self.state_in[0]: [c], # fake batch axis
                self.state_in[1]: [h],
        }
        if condition is not None:
            feed_dict.update({ self.c: [[condition]] })
        if z is not None:
            feed_dict.update({ self.z: [[z]] })
        return feed_dict

    def act(self, ob, ac, c, h, condition=None, z=None):
        sess = tf.get_default_session()

        feed_dict = self.get_feed_dict(ob, ac, c, h, condition, z)

        fetches = [self.one_hot_samples, self.vf] + self.state_out
        if not self.args.conditional:
            fetches += [self.z]

        out = sess.run(fetches, feed_dict)

        # TODO: need to extract one
        for idx, item in enumerate(out):
            if isinstance(item, dict):
                for name in item:
                    item[name] = item[name][0]
            else:
                item = item[0]
            out[idx] = item
        return out

    def decoder(self, z, action_sizes, data_format, lstm_size, scale=1):
        # [batch, max_time, lstm_size]
        z_shape = tf.shape(z)
        batch_size, max_time = z_shape[0], z_shape[1]

        one_hot_samples, samples, logits = {}, {}, {}

        for action_idx, (name, action_size) in enumerate(action_sizes.items()):
            # [batch*max_time, lstm_size]
            z_flat = tf.reshape(z, [-1, self.lstm_size])

            with tf.variable_scope("decoder_{}".format(name)):
                if len(action_size) == 1:
                    N = action_size[0]
                    logit = tl.dense(
                            z_flat, N,
                            activation=None,
                            name="action{}".format(name),
                            kernel_initializer= \
                                    normalized_columns_initializer(0.01))
                else:
                    # format: NHWC
                    reshape = tf.reshape(z_flat, [-1, 4, 4, int(lstm_size / 16)])

                    # format: NHWC
                    res = deconv = tl.conv2d_transpose(
                            reshape, int(32), 4,
                            strides=(2, 2),
                            padding='same',
                            activation=tf.nn.relu,
                            data_format='channels_last')

                    if data_format == 'channels_first' \
                            and args.dynamic_channel:
                        # format: NHWC -> NCHW
                        res = tf.transpose(res, [0, 3, 1, 2])

                    # format: each
                    for res_idx in range(int(8*scale)):
                        res = res_block(
                                res, int(32), 3, data_format,
                                name="decoder_res_{}".format(res_idx))

                    # format: NHWC
                    deconv = res
                    transposed = False
                    for deconv_idx in range(int(2)):
                        deconv_width = int(deconv.get_shape()[2])
                        if deconv_width == action_size[0]:
                            break

                        # format: NCHW -> NHWC 
                        if deconv_idx == 0 and data_format == 'channels_first' \
                                and args.dynamic_channel:
                            transposed = True
                            deconv = tf.transpose(deconv, [0, 2, 3, 1])

                        # format: NHWC
                        deconv = tl.conv2d_transpose(
                                deconv, int(32), 4,
                                strides=(2, 2),
                                padding='same',
                                activation=tf.nn.relu,
                                data_format='channels_last',
                                name="deconv_{}".format(deconv_idx))

                    # format: each
                    if data_format == 'channels_first' and transposed \
                            and args.dynamic_channel:
                        # format: NHWC -> NCHW
                        deconv = tf.transpose(deconv, [0, 3, 1, 2])

                    # format: each
                    conv = tl.conv2d(
                            deconv, 1, 3,
                            padding='same',
                            activation=None,
                            data_format=data_format,
                            name="conv_1x1")
                    
                    logit = tl.flatten(conv)

                logits[name] = tf.reshape(
                        logit, [batch_size, max_time, -1])

                action_one_hot, action = \
                        categorical_sample(logit, np.prod(action_size))

                # [batch, max_time, action_size[name]]
                one_hot_samples[name] = tf.reshape(
                        action_one_hot, [batch_size, max_time, -1],
                        name="one_hot_samples_{}".format(name))
                # [batch, max_time, 1]
                samples[name] = tf.reshape(
                        action, [batch_size, max_time],
                        name="samples_{}".format(name))

                if action_idx < len(action_sizes) - 1:
                    # this will be feeded to make gradient flows
                    out = mlp(
                            tf.expand_dims(samples[name], -1), int(16),
                            name='sample_mlp')
                    # [batch, max_time, lstm_size]
                    z = tl.dense(
                            tf.concat([z, out], -1), int(lstm_size),
                            activation=tf.nn.relu,
                            name="concat_z_fc")

        return one_hot_samples, samples, logits

    #def value(self, ob, c, h):
    #    sess = tf.get_default_session()
    #    feed_dict = {
    #            self.x: [[ob]],
    #            self.state_in[0]: c,
    #            self.state_in[1]: h,
    #    }
    #    return sess.run(self.vf, feed_dict)[0][0]

# TODO: not sure what this architecture is (1)
def mlp(x, dim, hid_dim=64, num_layers=3, name=None):
    for idx in range(num_layers-1):
        x = tl.dense(
                x, hid_dim,
                activation=tf.nn.relu,
                name="{}_{}".format(name, idx))
    x = tl.dense(
            x, dim,
            activation=tf.nn.relu,
            name="{}_{}".format(name, idx+1))
    return x

def res_block(x, channel, size, data_format, name):
    with tf.variable_scope(name):
        enc_x = tl.conv2d(
                x, channel, size,
                padding='same',
                activation=tf.nn.relu,
                data_format=data_format)

        res = tl.conv2d(
                enc_x, channel, size,
                padding='same',
                activation=None,
                data_format=data_format) + x
    return res

def categorical_sample(logits, d):
    out = tf.multinomial(logits - \
            tf.reduce_max(logits, [1], keepdims=True), 1)
    value = tf.squeeze(out, [1])
    return tf.one_hot(value, d), tf.cast(value, tf.float32)

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
