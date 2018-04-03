from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import models
import replay
import rl_utils
import utils as ut


logger = ut.logging.get_logger()

class Agent(object):

    def __init__(self, args, server, cluster, env, queue_shapes,
                 trajectory_queue_size, replay_queue_size):
        self.env = env
        self.args = args
        self.task = args.task
        self.queue_shapes = queue_shapes
        self.trajectory_queue_size = trajectory_queue_size
        self.replay_queue_size = replay_queue_size

        self.action_sizes = env.action_sizes
        self.input_shape = list(self.env.observation_shape)

        ###########################
        # Master policy (task!=1)
        ###########################

        device = 'gpu' if self.task < args.num_gpu else 'cpu'
        master_device = "/job:worker/task:{}/{}:0".format(self.args.task, device)
        master_device_replica = tf.train. \
                replica_device_setter(1, worker_device=master_device)

        with tf.device(master_device_replica):
            with tf.variable_scope("global"):
                self.policy_step = tf.get_variable(
                        "policy_step", [], tf.int32,
                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                        trainable=False)

            # master should initialize discriminator
            if args.task < 2:
                self.global_disc = models.Discriminator(
                        self.args, self.input_shape, "global")

        if args.task != 1 or args.loss == 'l2':
            logger.debug(master_device)

            with tf.device(master_device_replica):
                self.prepare_master_network()

        ##################################
        # Queue pipelines (ps/task=0~)
        ##################################
        with tf.device('/job:ps/task:0'):
            # TODO: we may need more than 1 queue
            #for i in range(cluster.num_tasks('ps')):
            if args.task != 1 or args.loss == 'l2':
                self.trajectory_queue = tf.FIFOQueue(
                        self.trajectory_queue_size,
                        [tf.float32] * len(self.queue_shapes),
                        shapes=[shape for _, shape in self.queue_shapes],
                        names=[name for name, _ in self.queue_shapes],
                        shared_name='queue')
                self.trajectory_queue_size_op = self.trajectory_queue.size()

            if args.loss == 'gan':
                self.replay_queue = tf.FIFOQueue(
                        self.replay_queue_size,
                        tf.float32,
                        shapes=dict(self.queue_shapes)['states'][1:],
                        shared_name='replay')
                self.replay_queue_size_op = self.replay_queue.size()
            else:
                self.replay_queue = None
                self.replay_queue_size_op = None

        ###########################
        # Master policy network
        ###########################
        if self.args.task == 0:
            policy_batch_size = self.args.policy_batch_size
            # XXX: may need this if you are lack of GPU memory
            #policy_batch_size = int(self.args.policy_batch_size \
            #        / self.env.episode_length)

            worker_device = "/job:worker/task:{}/cpu:0".format(self.task)
            logger.debug(worker_device)

            with tf.device(worker_device):
                with tf.variable_scope("global"):
                    self.trajectory_dequeue = self.trajectory_queue. \
                            dequeue_many(policy_batch_size)

        ###########################
        # Discriminator (task=1)
        ###########################
        elif self.args.task == 1 and self.args.loss == 'gan':
            if self.args.num_gpu > 1:
                gpu_num = 1
            else:
                gpu_num = 0

            worker_device = "/job:worker/task:{}/gpu:{}".format(self.task, gpu_num)
            logger.debug(worker_device)

            with tf.device(worker_device):
                self.prepare_gan()

            worker_device = "/job:worker/task:{}/cpu:0".format(self.task)
            logger.debug(worker_device)

            with tf.device(worker_device):
                with tf.variable_scope("global"):
                    self.replay_dequeue = self.replay_queue. \
                            dequeue_many(self.args.disc_batch_size)

        #####################################################
        # Local policy network (task >= 2 (gan) or 1 (l2))
        #####################################################
        elif self.args.task >= 1:
            worker_device = "/job:worker/task:{}/cpu:0".format(self.task)
            logger.debug(worker_device)

            with tf.device(worker_device):
                self.prepare_local_network()

    def prepare_master_network(self):
        self.global_network = pi = models.Policy(
                self.args, self.env, "global",
                self.input_shape, self.action_sizes,
                data_format='channels_first' \
                        if self.args.dynamic_channel \
                        else 'channels_last')

        self.acs, acs = {}, {}
        for idx, (name, action_size) in enumerate(
                self.action_sizes.items()):
            # [B, action_size]
            self.acs[name] = tf.placeholder(
                    tf.int32, [None, None], name="{}_in".format(name))
            acs[name] = tf.one_hot(self.acs[name], np.prod(action_size))

        self.adv = adv = tf.placeholder(
                tf.float32, [None, self.env.episode_length], name="adv")
        self.r = r = tf.placeholder(
                tf.float32, [None, self.env.episode_length], name="r")

        bsz = tf.to_float(tf.shape(pi.x)[0])

        ########################
        # Building optimizer
        ########################

        self.loss = 0
        self.pi_loss, self.vf_loss, self.entropy = 0, 0, 0

        for name in self.action_sizes:
            ac = acs[name]
            logit = pi.logits[name]

            log_prob_tf = tf.nn.log_softmax(logit)
            prob_tf = tf.nn.softmax(logit)

            pi_loss = - tf.reduce_sum(
                    tf.reduce_sum(log_prob_tf * ac, [-1]) * adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            self.loss += pi_loss + 0.5 * vf_loss - \
                    entropy * self.args.entropy_coeff

            self.pi_loss += pi_loss
            self.vf_loss += vf_loss
            self.entropy += entropy

        grads = tf.gradients(self.loss, pi.var_list)

        ##################
        # Summaries
        ##################

        # summarize only the last state
        summaries = [
                tf.summary.image("last_state", self.env.denorm(pi.x[:,-1])),
                tf.summary.scalar("env/r", tf.reduce_mean(self.r[:,-1])),
                tf.summary.scalar("model/policy_loss", self.pi_loss / bsz),
                tf.summary.scalar("model/value_loss", self.vf_loss / bsz),
                tf.summary.scalar("model/entropy", self.entropy / bsz),
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads)),
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list)),
        ]

        if pi.c is not None:
            summaries.append(
                    tf.summary.image("target", self.env.denorm(pi.c[:,-1])))

        self.summary_op = tf.summary.merge(summaries)
        grads, _ = tf.clip_by_global_norm(grads, self.args.grad_clip)

        grads_and_vars = list(zip(grads, self.global_network.var_list))

        # each worker has a different set of adam optimizer parameters
        opt = tf.train.AdamOptimizer(
                self.args.policy_lr, name="gan_optim")

        self.train_op = opt.apply_gradients(grads_and_vars, self.policy_step)
        self.summary_writer = None

    def prepare_local_network(self):
        self.local_network = models.Policy(
                self.args, self.env, "local",
                self.input_shape, self.action_sizes,
                data_format='channels_last')

        ##########################
        # Trajectory queue
        ##########################
        self.trajectory_placeholders = {
                name:tf.placeholder(
                        tf.float32, dict(self.queue_shapes)[name],
                        name="{}_in".format(name)) \
                                for name, shape in self.queue_shapes
        }
        self.trajectory_enqueues = self.trajectory_queue.enqueue(
                { name:self.trajectory_placeholders[name] \
                        for name, _ in self.queue_shapes })

        ##########################
        # Replay queue
        ##########################
        if self.args.loss == 'gan':
            self.replay_placeholder = tf.placeholder(
                    tf.float32, self.input_shape,
                    name="replay_in")
            self.replay_enqueue = self.replay_queue.enqueue(
                    self.replay_placeholder)
        else:
            self.replay_placeholder = None
            self.replay_enqueue = None

        ###############################
        # Thread dealing with queues
        ###############################
        self.worker_thread = rl_utils.WorkerThread(
                self.env,
                self.local_network,
                self.trajectory_enqueues,
                self.trajectory_placeholders,
                self.trajectory_queue_size_op,
                self.replay_enqueue,
                self.replay_placeholder,
                self.replay_queue_size_op)

        # copy weights from the parameter server to the local model
        self.policy_sync = ut.tf.get_sync_op(
                self.global_network.var_list,
                self.local_network.var_list)

    def prepare_gan(self):
        self.replay = replay.ReplayBuffer(self.args, self.input_shape)
        self.replay_dequeue = \
                self.replay_queue.dequeue_many(self.args.disc_batch_size)

        self.replay_thread = rl_utils.ReplayThread(
                self.replay, self.replay_dequeue)

        self.local_disc = models.Discriminator(
                self.args, self.input_shape, "local")

        self.disc_sync = ut.tf.get_sync_op(
                self.global_disc.var_list,
                self.local_disc.var_list)

    def start_worker_thread(self, sess, summary_writer):
        self.worker_thread.start_thread(sess, summary_writer)
        self.summary_writer = summary_writer

    def start_replay_thread(self, sess, summary_writer):
        self.replay_thread.start_thread(sess)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        rollout = self.worker_thread.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.worker_thread.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    ###########################
    # Master policy (task=0)
    ###########################

    def train_policy(self, sess):
        assert self.task < self.args.num_gpu, \
                "Only chief should use GPU and update"

        rollout = sess.run(self.trajectory_dequeue)
        batch = rl_utils.multiple_process_rollout(
                rollout, gamma=0.99, lambda_=1.0)

        #################
        # Feed ops
        #################

        feed_dict = {
                self.r: batch.r,
                self.adv: batch.adv,
                self.global_network.x: batch.si,
                self.global_network.ac: batch.a,
                self.global_network.state_in[0]: batch.features[:,0],
                self.global_network.state_in[1]: batch.features[:,1],
        }
        for name in self.action_sizes:
            name_a = batch.a[:,:,self.env.ac_idx[name]]
            feed_dict.update({
                    self.acs[name]: name_a,
            })
            if name in self.global_network.samples:
                feed_dict.update({
                        self.global_network.samples[name]: name_a,
                })
        if self.args.conditional:
            feed_dict.update({
                    self.global_network.c: batch.c,
            })

        #################
        # Fetch ops
        #################

        fetches = [
                self.train_op, self.summary_op, self.policy_step,
                self.trajectory_queue_size_op,
        ]

        if self.args.debug:
            var_to_test = self.global_network.var_list
            before = sess.run(var_to_test)

        #if batch.r.sum() == 1:
        #    import ipdb; ipdb.set_trace() 
        out = sess.run(fetches, feed_dict=feed_dict)

        if self.args.debug:
            after = sess.run(var_to_test)
            weights_before_after(before, after, var_to_test)

        self.summary_writer.add_summary(
                tf.Summary.FromString(out[1]), out[2])
        self.summary_writer.flush()

        debug_text = "# traj: {}".format(out[3])
        if self.task == 0:
            logger.info(debug_text)
        else:
            logger.debug(debug_text)

    ###########################
    # Discriminator (task=1)
    ###########################

    def train_gan(self, sess):
        fakes = self.replay.sample(self.args.fake_batch_size)

        feed_dict = {
                self.disc.fake: fakes,
                self.disc.real: self.env.get_random_target(self.args.real_batch_size),
        }
        fetches = [
                self.disc.train_op, self.disc.summary_op, self.disc.step,
        ]
        out = sess.run(fetches, feed_dict=feed_dict)

        self.summary_writer.add_summary(tf.Summary.FromString(out[1]), out[2])
        self.summary_writer.flush()


def weights_before_after(before, after, var_to_test):
    print(" [*] Weight change check")

    for idx, (bef, aft, var) in \
            enumerate(zip(before, after, var_to_test)):
        assert bef.shape == aft.shape, \
                "Shape [{}] is not same: {}, {}".format(
                        var.name, bef.shape, aft.shape)

        bef_sum, aft_sum = bef.sum(), aft.sum()
        same_or_not = "SAME" if bef_sum == aft_sum else "    "

        print(" [{}] {}: {} ({}, {})". \
            format(idx, var.name, same_or_not, bef_sum, aft_sum))

