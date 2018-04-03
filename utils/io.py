# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import json
import math
import timeit
import numpy as np
import cloudpickle
from six.moves import shlex_quote

from . import logging

logger = logging.get_logger()

try:
    import scipy.misc
    imread = scipy.misc.imread
    imresize = scipy.misc.imresize
    imsave = imwrite = scipy.misc.imsave
except:
    import cv2
    imread = cv2.imread
    imresize = cv2.imresize
    imsave = imwrite = cv2.imwrite


def get_cmd(as_list=False):
    args = [shlex_quote(arg) for arg in sys.argv][1:]
    if as_list:
        return args
    return ' '.join(args)

class Timer:
    """Example:
        with ut.io.Timer("tokenize"):
            for text, score in ut.io.read_csv(path):
                chars = korean.tokenize(text) + ['<eos>']
                for char in chars:
                    self.dictionary.add_char(char)
    """
    def __init__(self, desc=""):
        self.desc = desc

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.interval = self.end - self.start
        logger.debug(f"[tt] {self.desc}: {self.interval:.3f}s")

def makedirs(path):
    path = str(path)
    if not os.path.exists(path):
        logger.info(f"Make directories: {path}")
        os.makedirs(path)
    else:
        logger.warning(f"Skip making directories: {path}")

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Removed: {path}")

#####################
# Pickle
#####################

def _dump_pickle(path, data):
    path = str(path)
    with open(path, 'wb') as f, Timer(f"Dumped pickle: {path}"):
        cloudpickle.dump(data, f)

def dump_pickle(path, data, num_split=1):
    if num_split == 1:
        _dump_pickle(path, data)
    else:
        pivot = 0
        window_size = math.ceil(len(data) / num_split)
        for idx in trange(num_split, desc=f"Dump {num_split} pickles"):
            new_path = add_postfix(path, idx)
            _dump_pickle(new_path, data[pivot:pivot+window_size])
            pivot += window_size

def _load_pickle(path):
    path = str(path)
    with open(path, 'rb') as f, Timer(f"Loaded pickle: {path}"):
        data = cloudpickle.load(f)
    return data

def load_pickle(path, num_split=1):
    if num_split == 1:
        data = _load_pickle(path)
    else:
        data = []
        for idx in trange(num_split, desc=f"Load {num_split} pickles"):
            new_path = add_postfix(path, idx)
            tmp_data = _load_pickle(new_path)
            data.extend(tmp_data)
    return data

