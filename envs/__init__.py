# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .simple import Simple
from .mnist import MNIST, SimpleMNIST


def create_env(args):
    env = args.env.lower()
    if env == 'simple':
        env = Simple(args)
    elif env == 'simple_mnist':
        env = SimpleMNIST(args)
    elif env == 'mnist':
        env = MNIST(args)
    else:
        raise Exception(f"Unkown environment: {args.env}")
    return env
