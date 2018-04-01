# -*- coding: future_fstrings -*-
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

        worker_device = f"/job:worker/task:{self.task}" \
                        f"/{'gpu' if self.task < args.num_gpu else 'cpu'}:0"

        ###########################
        # Master policy (task=0)
        ###########################
        master_device = tf.train. \
                replica_device_setter(1, worker_device=worker_device)

        with tf.device(master_device):
            self.prepare_master_network()

        ##################################
        # Queue pipelines (ps/task=0~)
        ##################################
        with tf.device('/job:ps/task:0'):
            # TODO: we may need more than 1 queue
            #for i in range(cluster.num_tasks('ps')):
            self.trajectory_queue = tf.FIFOQueue(
                    self.trajectory_queue_size,
                    [tf.float32] * len(self.queue_shapes),
                    shapes=[shape for _, shape in self.queue_shapes],
                    names=[name for name, _ in self.queue_shapes],
                    shared_name='queue')
            self.trajectory_queue_size_op = self.trajectory_queue.size()

            self.replay_queue = tf.FIFOQueue(
                    self.replay_queue_size,
                    tf.float32,
                    shapes=dict(self.queue_shapes)['states'],
                    shared_name='replay')
            self.replay_queue_size_op = self.replay_queue.size()

        ###########################
        # Discriminator (task=1)
        ###########################
        if self.args.task == 0:
            policy_batch_size = self.args.policy_batch_size
            # XXX: may need this if you are lack of GPU memory
            #policy_batch_size = int(self.args.policy_batch_size \
            #        / self.env.episode_length)

            worker_device = f"/job:worker/task:{self.task}/cpu:0"
            with tf.device(worker_device):
                with tf.variable_scope("global"):
                    self.dequeues = self.trajectory_queue. \
                            dequeue_many(policy_batch_size)
        elif self.args.task == 1 and self.args.loss == 'gan':
            self.prepare_gan()
        #####################################################
        # Local policy network (task >= 2 (gan) or 1 (l2))
        #####################################################
        elif self.args.task >= 1:
            worker_device = f"/job:worker/task:{self.task}/cpu:0"
            with tf.device(worker_device):
                self.prepare_local_network()

    def prepare_master_network(self):
        with tf.variable_scope("global"):
            self.policy_step = tf.get_variable(
                    "policy_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)

            self.global_network = pi = models.Policy(
                    self.args, self.env,
                    self.input_shape, self.action_sizes,
                    data_format='channels_first' \
                            if self.args.dynamic_channel \
                            else 'channels_last')

        self.acs, acs = {}, {}
        for idx, (name, action_size) in enumerate(
                self.action_sizes.items()):
            # [B, action_size]
            self.acs[name] = tf.placeholder(
                    tf.int32, [None, None], name=f"{name}_in")
            acs[name] = tf.one_hot(self.acs[name], np.prod(action_size))

        self.adv = adv = tf.placeholder(
                tf.float32, [None, self.env.episode_length], name="adv")
        self.r = r = tf.placeholder(
                tf.float32, [None, self.env.episode_length], name="r")

        self.loss = 0
        bsz = tf.to_float(tf.shape(pi.x)[0])

        for name in self.action_sizes:
            ac = acs[name]
            self.logit = pi.logits[name]

            log_prob_tf = tf.nn.log_softmax(self.logit)
            prob_tf = tf.nn.softmax(self.logit)

            pi_loss = tf.reduce_sum(
                    tf.reduce_sum(log_prob_tf * ac, [-1]) * adv)

            # loss of value function
            self.vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            self.loss += pi_loss + 0.5 * self.vf_loss \
                    - entropy * self.args.entropy_coeff

        grads = tf.gradients(self.loss, pi.var_list)

        # summarize only the last state
        tf.summary.image("last_state", self.env.denorm(pi.x[:,-1]))
        if pi.c is not None:
            tf.summary.image("target", self.env.denorm(pi.c[:,-1]))
        tf.summary.scalar("env/r", tf.reduce_mean(self.r[:,-1]))
        tf.summary.scalar("model/policy_loss", pi_loss / bsz)
        tf.summary.scalar("model/value_loss", self.vf_loss / bsz)
        tf.summary.scalar("model/entropy", entropy / bsz)
        tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))

        self.summary_op = tf.summary.merge_all()
        grads, _ = tf.clip_by_global_norm(grads, self.args.grad_clip)

        grads_and_vars = list(zip(grads, self.global_network.var_list))
        policy_inc_step = self.policy_step.assign_add(tf.shape(pi.x)[0])

        # each worker has a different set of adam optimizer parameters
        opt = tf.train.AdamOptimizer(self.args.policy_lr)

        self.train_op = tf.group(
                opt.apply_gradients(grads_and_vars), policy_inc_step)
        self.summary_writer = None

    def prepare_local_network(self):
        with tf.variable_scope("local"):
            self.local_network = models.Policy(
                    self.args, self.env,
                    self.input_shape, self.action_sizes,
                    data_format='channels_last')

            ##########################
            # Trajectory queue
            ##########################
            self.trajectory_placeholders = {
                    name:tf.placeholder(
                            tf.float32, dict(self.queue_shapes)[name],
                            name=f"{name}_in") \
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
                        tf.float32, [None] + self.input_shape,
                        name=f"replay_in")
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
        self.sync = ut.tf.get_sync_op(
                self.global_network.var_list,
                self.local_network.var_list)

    def prepare_gan(self):
        if self.args.num_gpu == 2:
            worker_device = f"/job:worker/task:{self.task}/gpu:1"
        else:
            worker_device = f"/job:worker/task:{self.task}/gpu:0"

        self.replay = replay.ReplayBuffer(self.args, self.input_shape)
        self.replay_dequeue = \
                self.replay_queue.dequeue_many(self.args.discrim_batch_size)

        self.replay_thread = rl_utils.ReplayThread(
                self.replay, self.replay_dequeue)

        with tf.device(worker_device):
            self.discrim_step = tf.get_variable(
                    "discrim_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)

            self.discrim = models.Discriminator(
                    self.args, self.input_shape)

            grads_and_vars = list(zip(grads, self.discrim.var_list))
            discrim_inc_step = self.policy_step.assign_add(tf.shape(pi.x)[0])

            discrim_opt = tf.train.AdamOptimizer(self.args.policy_lr)
            self.discrim_optim = tf.group(
                    discrim_opt.apply_gradients(grads_and_vars),
                    discrim_inc_step)

    def start_worker_thread(self, sess, summary_writer):
        self.worker_thread.start_thread(sess, summary_writer)
        self.summary_writer = summary_writer

    def start_replay_thread(self, sess, summary_writer):
        self.worker_thread.start_thread(sess, summary_writer)
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

        rollout = sess.run(self.dequeues)
        batch = rl_utils.multiple_process_rollout(
                rollout, gamma=0.99, lambda_=1.0)

        feed_dict = {
                self.r: batch.r,
                self.adv: batch.adv,
                self.global_network.x: batch.si,
                self.global_network.ac: batch.a,
                self.global_network.state_in[0]: batch.features[:,0],
                self.global_network.state_in[1]: batch.features[:,1],
        }
        for name in self.action_sizes:
            feed_dict.update({
                    self.acs[name]: batch.a[:,:,self.env.ac_idx[name]],
            })
        if self.args.conditional:
            feed_dict.update({
                    self.global_network.c: batch.c,
            })
        fetches = [
                self.train_op, self.summary_op, self.policy_step,
                self.trajectory_queue_size_op,
                self.replay_queue_size_op,
        ]
        out = sess.run(fetches, feed_dict=feed_dict)

        self.summary_writer.add_summary(
                tf.Summary.FromString(out[1]), out[2])
        self.summary_writer.flush()

        if self.task == 0:
            logger.info(f"# traj: {out[3]}, # replay: {out[4]}")
        else:
            logger.debug(f"# traj: {out[3]}, # replay: {out[4]}")

    ###########################
    # Discriminator (task=1)
    ###########################

    def train_gan(self, sess):
        pass

