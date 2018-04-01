# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys, signal
import tensorflow as tf

import utils as ut
from agent import Agent
from envs import create_env

logger = ut.logging.get_logger()


def train(args, server, cluster, env, queue_shapes,
          trajectory_queue_size, replay_queue_size):

    agent = Agent(args, server, cluster, env, queue_shapes,
                  trajectory_queue_size, replay_queue_size)

    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [
            v for v in tf.global_variables() if not v.name.startswith("local")]

    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = ut.tf.FastSaver(variables_to_save)

    var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    devices = ["/job:ps"]
    if args.task == 0:
        devices += [f"/job:worker/task:{args.task}/gpu:0", f"/job:worker/task:{args.task}/cpu:0"]
    else:
        devices += [f"/job:worker/task:{args.task}/cpu:0"]

    config = tf.ConfigProto(device_filters=devices, allow_soft_placement=True)
    logger.info("Events directory: %s_%s", args.load_path, args.task)

    summary_writer = tf.summary.FileWriter(f"{args.load_path}_{args.task}")
    agent.summary_writer = summary_writer

    sv = tf.train.Supervisor(
            is_chief=args.task == 0,
            logdir=str(args.load_path),
            saver=saver,
            summary_op=None,
            init_op=init_op,
            init_fn=init_fn,
            summary_writer=summary_writer,
            ready_op=tf.report_uninitialized_variables(variables_to_save),
            global_step=agent.policy_step,
            save_model_secs=30,
            save_summaries_secs=30)

    num_policy_steps = 100000000

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting"
        " to connect to the parameter server. One common cause is that"
        " the parameter server DNS name isn't resolving yet, or is misspecified.")

    with sv.managed_session(server.target, config=config) as sess, \
            sess.as_default():

        def sync():
            #logger.error("SYNC")
            sess.run(agent.sync)

        ###############################
        # Run thread
        ###############################
        if args.task == 1 and args.loss == 'gan':
            agent.start_replay_thread(sess)
        elif args.task >= 1:
            sync()
            agent.start_worker_thread(sess, summary_writer)

        policy_step = sess.run(agent.policy_step)
        logger.info("Starting training at step=%d", policy_step)

        while not sv.should_stop() and ( \
                not num_policy_steps or policy_step < num_policy_steps):
            if args.task == 0:
                agent.train_policy(sess)
            elif args.task == 1 and args.loss == 'gan':
                agent.train_gan(sess)
            else:
                sync()
            policy_step = sess.run(agent.policy_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', policy_step)

