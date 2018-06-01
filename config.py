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
model_arg.add_argument('--lstm_size', default=256, type=int)
model_arg.add_argument('--scale', default=1.0, type=float)
model_arg.add_argument('--z_dim', default=10, type=int)
model_arg.add_argument('--dynamic_channel', default=False, type=ut.args.str2bool)
model_arg.add_argument('--disc_dim', default=64, type=int)
model_arg.add_argument('--disc_batch_norm', default=True, type=ut.args.str2bool)
model_arg.add_argument('--loss', default='gan', type=str,
                       choices=['l2', 'gan'])


# environment
env_arg = ut.args.add_argument_group(parser, 'environment')
env_arg.add_argument('--env', default="simple_mnist")
env_arg.add_argument('--jump', default=True, type=ut.args.str2bool)
env_arg.add_argument('--curve', default=True, type=ut.args.str2bool)
env_arg.add_argument('--episode_length', default=5, type=int)
env_arg.add_argument('--screen_size', default=64, type=int)
env_arg.add_argument('--location_size', default=32, type=int)
env_arg.add_argument('--color_channel', default=3, type=int, choices=[3, 1])
env_arg.add_argument('--mnist_nums', default='0,1,2,3,4,5,6,7,8,9', type=ut.args.int_list)
env_arg.add_argument('--brush_path', default='assets/brushes/dry_brush.myb', type=str)
env_arg.add_argument('--conditional', default=True, type=ut.args.str2bool)


# train
train_arg = ut.args.add_argument_group(parser, 'train')
train_arg.add_argument('--policy_lr', default=1e-5, type=float)
train_arg.add_argument('--disc_lr', default=1e-4, type=float)
train_arg.add_argument('--clip_disc_weights', default=False, type=ut.args.str2bool)
train_arg.add_argument('--entropy_coeff', default=0.01, type=float)
train_arg.add_argument('--grad_clip', default=40, type=int)
train_arg.add_argument('--policy_batch_size', default=64, type=int)
train_arg.add_argument('--disc_batch_size', default=64, type=int)
train_arg.add_argument('--replay_size', default=10, type=int)
train_arg.add_argument('--buffer_batch_num', default=20, type=int)
train_arg.add_argument('--wgan_lambda', default=20, type=float)
train_arg.add_argument('--train', default=True, type=ut.args.str2bool)


# distributed
dist_arg = ut.args.add_argument_group(parser, 'distributed')
dist_arg.add_argument('--task', default=0, type=int)
dist_arg.add_argument('--job_name', default="worker")
dist_arg.add_argument('--num_workers', default=4, type=int)
dist_arg.add_argument('--start_port', default=13333, type=int)
dist_arg.add_argument('--tag', default='spiral', type=str)


# Misc
misc_arg = ut.args.add_argument_group(parser, 'misc')
misc_arg.add_argument('--debug', type=ut.args.str2bool, default=False)
misc_arg.add_argument('--num_gpu', type=int, default=1,
                      choices=[0, 1, 2])
misc_arg.add_argument('--policy_log_step', type=int, default=20)
misc_arg.add_argument('--disc_log_step', type=int, default=50)
misc_arg.add_argument('--data_dir', type=Path, default='.data')
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
    if args.loss == 'gan':
        args.conditional = False
        assert args.num_workers > 2, "num_workers should be larger than 2 (policy, discriminator, worker)"
    elif args.loss == 'l2':
        args.conditional = True
        assert args.num_workers > 1, "num_workers should be larger than 2 (policy, worker)"

    if parse_unknown:
        return args, unknown
    return args
