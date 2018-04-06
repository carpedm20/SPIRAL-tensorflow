from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import numpy as np
import scipy.signal
from collections import namedtuple

import utils as ut


logger = ut.logging.get_logger()

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "features", "c", "z"])


def discount(x, gamma):
    return scipy.signal.lfilter(
            [1], [1, -gamma], x[:,::-1], axis=1)[:,::-1]

def flatten_first_two(x):
    return np.reshape(x, [-1] + list(x.shape)[2:])

def multiple_process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout['states'])
    batch_a = np.asarray(rollout['actions'])
    rewards = np.asarray(rollout['rewards'])
    vpred_t = np.hstack(
            [rollout['values'][:,:,0], np.expand_dims(rollout['r'], -1)])

    rewards_plus_v = np.hstack(
            [rollout['rewards'], np.expand_dims(rollout['r'], -1)])
    batch_r = discount(rewards_plus_v, gamma)[:,:-1]
    delta_t = rewards + gamma * vpred_t[:,1:] - vpred_t[:,:-1]

    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout['features'][:,0]

    if 'conditions' in rollout:
        batch_c = np.asarray(rollout['conditions'])
        batch_z = None
    else:
        batch_c = None
        batch_z = np.asarray(rollout['z'])

    #batch_a = flatten_first_two(batch_a)
    #batch_r = flatten_first_two(batch_r)
    #batch_si = flatten_first_two(batch_si)
    #batch_adv = flatten_first_two(batch_adv)
    #features = features[:,:,0,:]

    return Batch(batch_si, batch_a, batch_adv, batch_r, features, batch_c, batch_z)


class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.features = []
        self.conditions = None
        self.z = None

    def add(self, state, action, reward, value, features, conditions=None, z=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.features += [features]

        if conditions is not None:
            if self.conditions is None:
                self.conditions = []
            self.conditions += [conditions]

        if z is not None:
            if self.z is None:
                self.z = []
            self.z += [z]


class WorkerThread(threading.Thread):
    def __init__(self, env, policy,
                 traj_enqueues, traj_placeholders, traj_size,
                 replay_enqueue, replay_placeholder, replay_size):
        threading.Thread.__init__(self)

        self.env = env
        self.sess = None
        self.daemon = True
        self.policy = policy
        self.last_features = None
        self.summary_writer = None
        self.num_local_steps = env.episode_length

        self.traj_enqueues = traj_enqueues
        self.traj_placeholders = traj_placeholders
        self.traj_size = traj_size

        self.replay_enqueue = replay_enqueue
        self.replay_placeholder = replay_placeholder
        self.replay_size = replay_size

    def start_thread(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(
                self.env, self.policy,
                self.num_local_steps, self.summary_writer)
        while True:
            out = next(rollout_provider)

            feed_dict = {
                    self.traj_placeholders['actions']: out.actions,
                    self.traj_placeholders['states']: out.states,
                    self.traj_placeholders['rewards']: out.rewards,
                    self.traj_placeholders['values']: out.values,
                    self.traj_placeholders['features']: out.features,
                    self.traj_placeholders['r']: out.r,
            }
            if self.env.conditional:
                feed_dict.update({
                        self.traj_placeholders['conditions']: out.conditions,
                })
            else:
                feed_dict.update({
                        self.traj_placeholders['z']: out.z,
                })

            for k, v in feed_dict.items():
                if isinstance(v, list):
                    feed_dict[k] = np.array(v)

            fetches = [
                    self.traj_enqueues,
            ]
            if self.replay_enqueue is not None:
                fetches.append(self.replay_enqueue)
                feed_dict.update({
                        self.replay_placeholder: out.states[-1],
                })

            out = self.sess.run(fetches, feed_dict)


class ReplayThread(threading.Thread):
    def __init__(self, replay, replay_dequeue):
        threading.Thread.__init__(self)

        self.replay = replay
        self.replay_dequeue = replay_dequeue

    def start_thread(self, sess):
        self.sess = sess
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        while True:
            generated = self.sess.run(self.replay_dequeue)
            self.replay.push(generated)


def env_runner(env, policy, num_local_steps, summary_writer):
    last_state, condition, z = env.reset()
    last_features = policy.get_initial_features(1, flat=True)

    length = 0
    rewards = 0

    while True:
        rollout = PartialRollout()

        last_action = env.initial_action

        for _ in range(num_local_steps):
            c, h = last_features

            fetched = policy.act(
                    last_state, last_action, c, h, condition, z)
            action, value_, features = fetched[0], fetched[1], fetched[2:4]

            action = [np.argmax(action[name]) for name in env.acs]
            state, reward, terminal, info = env.step(action)

            # collect the experience
            rollout.add(last_state, action, reward,
                        value_, last_features, condition, z)
            length += 1

            # TODO: discriminator communication to get reward
            rewards += reward

            last_state = state
            last_action = action
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

        last_state, condition, z = env.reset()
        logger.debug(
                "Episode finished. Sum of rewards: {:.5f}." \
                "Length: {}.".format(rewards, length))

        length = 0
        rewards = 0

        rollout.states += [state]

        # once we have enough experience, yield it,
        # and have the ThreadRunner place it on a queue
        yield rollout
