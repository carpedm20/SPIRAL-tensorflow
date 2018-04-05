# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.client import device_lib


# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def get_all_variables(scope_name):
     return tf.get_collection(
             tf.GraphKeys.GLOBAL_VARIABLES,
             scope=scope_name)

def get_sync_op(from_list, to_list):
    assert len(from_list) == len(to_list), \
            "length of to variables should be same ({len(from_list)} != {len(to_list)})"
    syncs = []
    for from_v, to_v in zip(from_list, to_list):
        assert from_v.get_shape() == to_v.get_shape(), \
                f"{from_v.get_shape()} != {to_v.get_shape()}" \
                f" ({from_v.name}, {to_v.name})"
        sync = to_v.assign(from_v)
        syncs.append(sync)
    return tf.group(*syncs)

def cluster_spec(num_workers, num_ps, port=12222):
    cluster = {}

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]
