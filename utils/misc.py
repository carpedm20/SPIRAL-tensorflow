# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import uuid
from datetime import datetime
from collections import defaultdict
from tensorflow.python.client import device_lib


def count_gpu():
    devices = device_lib.list_local_devices()
    return sum(1 for d in devices if d.device_type == 'GPU')

def get_hash(length):
    assert length <= 32, "length of hash should be less than 32"
    hash_text = uuid.uuid4().hex
    return hash_text[:length]

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f'[{status}] {bar}| [{count}/{total}] {percents}%\r')
    sys.stdout.flush() 


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret
