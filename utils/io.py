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
from six.moves import shlex_quote

from . import logging

logger = logging.get_logger()


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
    if not os.path.exists(path):
        logger.info(f"Make directories: {path}")
        os.makedirs(path)
    else:
        logger.warning(f"Skip making directories: {path}")

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Removed: {path}")

