# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ReplayBuffer(object):
    # Code based on https://github.com/carpedm20/simulated-unsupervised-tensorflow/blob/master/replay.py
    def __init__(self, args, observation_shape):
        self.rng = np.random.RandomState(args.seed)
        self.replay_size = args.replay_size
        self.batch_size = args.disc_batch_size

        self.idx = 0
        replay_shape = [self.replay_size] + observation_shape
        self.data = np.zeros(replay_shape)

    def push(self, batches):
        batch_size = len(batches)
        if self.idx + batch_size > self.replay_size:
            random_idx1 = self.rng.choice(self.idx, self.batch_size/2)
            random_idx2 = self.rng.choice(batch_size, self.batch_size/2)
            self.data[random_idx1] = batches[random_idx2]
        else:
            self.data[self.idx:self.idx+batch_size] = batches
            self.idx += batch_size

    def sample(self, n=None):
        if n is None:
            n = self.batch_size/2
        while self.idx < n:
            pass
        random_idx = self.rng.choice(self.idx, n)
        return self.data[random_idx]
