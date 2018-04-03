# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random
import numpy as np
from pathlib import Path

from . import io
from . import misc
from . import logging

PARAM_FNAME = "params.json"


logger = logging.get_logger()

def set_global_seed(seed, tensorflow=False, tf=False, pytorch=False):
    if tf or tensorflow:
        try:
            import tensorflow as tf
        except ImportError:
            pass
        else:
            tf.set_random_seed(seed)

    if pytorch:
        try:
            import torch as th
        except ImportError:
            pass
        else:
            th.manual_seed(seed)
            if th.cuda.is_available():
                th.cuda.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

def prepare_dirs(args):
    if args.load_path:
        if str(args.load_path).startswith(str(args.log_dir)):
            load_path = Path(args.load_path)
        else:
            load_path = Path(f"{str(args.log_dir)}/{args.load_path}")
        args.model_name = load_path.name
    else:
        model_desc = io.get_cmd().replace(" ", "|")
        hash_text = misc.get_hash(6)
        args.hash_text = hash_text

        args.model_name = f"{args.env}_{misc.get_time()}{model_desc}_{hash_text}"
        load_path = args.log_dir / args.model_name

    args.load_path = load_path
    io.makedirs(args.load_path)

    # create directories
    for key, path in vars(args).items():
        if key.endswith('_dir') and not os.path.exists(str(path)):
            io.makedirs(path)

def save_args(args):
    load_path = Path(args.load_path)
    param_path = load_path / PARAM_FNAME

    info = { k:str(v) if isinstance(v, Path) else v  for k, v in args.__dict__.items() }
    with open(str(param_path), 'w') as f:
        json.dump(info, f, indent=4, sort_keys=True)

    cmd_path = load_path / "cmd.sh"
    with open(str(cmd_path), 'w') as f:
        f.write(io.get_cmd())

    logger.info(f"Saved {PARAM_FNAME}: {param_path}")

def update_args(args, key, new_value):
    load_path = Path(args.load_path)
    param_path = load_path / PARAM_FNAME

    if param_path.exists():
        with open(param_path) as f:
            saved_args = json.load(f)

        original_value = saved_args.get(key, None)
        saved_args[key] = new_value

        with open(param_path, 'w') as f:
            json.dump(saved_args, f, indent=4, sort_keys=True)

        logger.info(f"Update {param_path}: {key} ({original_value} -> {new_value})")
    else:
        raise FileNotFoundError(f"{param_path} not exists")

# XXX: actually `skip_list` is quite important during test time
def load_args(args, skip_list=['load_path', 'test_epoch', 'test_dataset', 'train']):
    args_keys = vars(args).keys()
    args_path = os.path.join(args.load_path, PARAM_FNAME)

    with open(args_path) as f:
        saved_args = json.load(f)

    for saved_key, saved_value in saved_args.items():
        if saved_key in skip_list:
            continue

        if hasattr(args, saved_key):
            args_value = getattr(args, saved_key)
            if args_value != saved_value:
                if isinstance(args_value, Path):
                    # we don't have to print this
                    saved_value = Path(saved_value)
                else:
                    logger.info(f"[UPDATE] args `{saved_key}`: {args_value} -> {saved_value}")
                setattr(args, saved_key, saved_value)
        else: # if key is programmatically generated, skip and let the code make the key by itself
            pass

