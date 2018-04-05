# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ReplayBuffer(object):
    # Code based on https://github.com/carpedm20/simulated-unsupervised-tensorflow/blob/master/replay.py
    def __init__(self, args, observation_shape):
        self.replay_size = args.replay_size
        self.buffer_size = args.buffer_size
        self.batch_size = args.disc_batch_size

        self.rng = np.random.RandomState(args.seed)

        self.idx = 0
        replay_shape = [self.buffer_size] + observation_shape
        self.data = np.zeros(replay_shape, dtype=np.uint8)

        self.most_recent = None

    def push(self, batches):
        batch_size = len(batches)
        if self.idx + batch_size > self.buffer_size:
            random_idx1 = self.rng.choice(self.idx, int(self.batch_size/2))
            random_idx2 = self.rng.choice(batch_size, int(self.batch_size/2))
            self.data[random_idx1] = batches[random_idx2]
        else:
            self.data[self.idx:self.idx+batch_size] = batches
            self.idx += int(batch_size)

        self.most_recent = batches

    def sample(self, n):
        while self.idx < n:
            pass

        replay_idx = self.rng.choice(
                self.idx, self.replay_size)
        replay = self.data[replay_idx].astype(np.float32)

        recent_idx = self.rng.choice(
                len(self.most_recent), self.batch_size - self.replay_size)
        recent = self.most_recent[recent_idx].astype(np.float32)

        return np.concatenate([replay, recent], 0)
