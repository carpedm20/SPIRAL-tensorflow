# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from pathlib import Path

import utils as ut

logger = ut.logging.get_logger()


parser = argparse.ArgumentParser()

# model
model_arg = ut.args.add_argument_group(parser, 'model')
model_arg.add_argument('--lstm-size', default=256, type=int)
model_arg.add_argument('--scale', default=1.0, type=float)
model_arg.add_argument('--z_dim', default=100, type=int)
model_arg.add_argument('--conditional', default=True, type=ut.args.str2bool)
model_arg.add_argument('--dynamic_channel', default=False, type=ut.args.str2bool)


# environment
env_arg = ut.args.add_argument_group(parser, 'environment')
env_arg.add_argument('--env', default="simple")
env_arg.add_argument('--episode_length', type=int, default=5)
env_arg.add_argument('--screen_size', type=int, default=64)
env_arg.add_argument('--location_size', type=int, default=8)
env_arg.add_argument('--color_channel', type=int, default=3, choices=[3, 1])


# train
train_arg = ut.args.add_argument_group(parser, 'train')
train_arg.add_argument('--loss', default='gan', type=str,
                       choices=['l2', 'gan'])
train_arg.add_argument('--policy_lr', default=1e-4, type=float)
train_arg.add_argument('--disc_lr', default=1e-5, type=float)
train_arg.add_argument('--entropy_coeff', default=0.01, type=float)
train_arg.add_argument('--grad_clip', default=40, type=int)
train_arg.add_argument('--policy_batch_size', default=64, type=int)
train_arg.add_argument('--real_batch_size', default=44, type=int)
train_arg.add_argument('--fake_batch_size', default=20, type=int)
#train_arg.add_argument('--discrim_batch_size') = real_batch_size + fake_batch_size
train_arg.add_argument('--replay_size', default=64000, type=int)


# distributed
dist_arg = ut.args.add_argument_group(parser, 'distributed')
dist_arg.add_argument('--task', default=0, type=int)
dist_arg.add_argument('--job_name', default="worker")
dist_arg.add_argument('--num_workers', default=4, type=int)
dist_arg.add_argument('--start_port', default=13333, type=int)


# Misc
misc_arg = ut.args.add_argument_group(parser, 'misc')
misc_arg.add_argument('--num_gpu', type=int, default=1,
                      choices=[1, 2])
misc_arg.add_argument('--log_dir', type=Path, default='logs')
misc_arg.add_argument('--load_path', type=Path, default=None)
misc_arg.add_argument('--log_level', type=str, default='INFO',
                      choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--seed', type=int, default=123)
misc_arg.add_argument('--dry_run', action='store_true')
misc_arg.add_argument('--tb_port', type=int, default=12345)


def get_args(group_name=None, parse_unknown=False):
    if parse_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()

    ##############################
    # Preprocess or filter args
    ##############################
    args.discrim_batch_size = \
            args.real_batch_size + args.fake_batch_size

    assert args.num_workers > 2, "num_workers should be larger than 2 (policy, discriminator, worker)"

    if parse_unknown:
        return args, unknown
    return args
