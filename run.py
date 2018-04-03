# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
from six.moves import shlex_quote

import utils as ut


def new_cmd(session, name, cmd, load_path, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)
    return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))


def create_commands(session, args, shell='bash'):
    ut.train.prepare_dirs(args)

    actual_args = ut.io.get_cmd(as_list=True)
    actual_cmd = ' '.join(actual_args)

    # for launching the TF workers and for launching tensorboard
    base_cmd = [
            'CUDA_VISIBLE_DEVICES=',
            sys.executable, 'main.py',
            '--load_path', args.load_path,
            '--start_port', args.start_port,
            '--num_gpu', ut.misc.count_gpu(),
    ] + actual_args

    cmds_map = [new_cmd(session, "ps", base_cmd + ["--job_name", "ps"], args.load_path, shell)]

    if args.loss == 'l2':
        gpu_task_num = 1
    elif args.loss == 'gan':
        gpu_task_num = 2

    for i in range(args.num_workers):
        if i < gpu_task_num: # gpu workers
            cmd = base_cmd[1:]
        else:
            cmd = base_cmd[:]

        cmd += ["--job_name", "worker", "--task", str(i)]
        cmds_map += [new_cmd(session, "w-%d" % i, cmd, args.load_path, shell)]

    tmp_tb_dir = "/".join(sys.executable.split('/')[:-1])
    tmp_tb_path = os.path.join(tmp_tb_dir, "tensorboard")

    if os.path.exists(tmp_tb_path):
        tb = tmp_tb_dir + "/tensorboard"
    else:
        tb = "tensorboard"
    tb_args = [tb, "--logdir", args.log_dir, "--port", "12345"]

    cmds_map += [new_cmd(session, "tb", tb_args, args.load_path, shell)]
    cmds_map += [new_cmd(session, "htop", ["htop"], args.load_path, shell)]

    windows = [v[0] for v in cmds_map]

    notes = []
    cmds = []

    notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
    notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]

    notes += ["Point your browser to http://localhost:12345 to see Tensorboard"]

    cmds += [
            # kill any process using tensorboard's port
            f"kill $( lsof -i:{args.tb_port} -t ) > /dev/null 2>&1",
            # kill any processes using ps / worker ports
            f"kill $( lsof -i:{args.start_port}-{args.num_workers + args.start_port} -t ) > /dev/null 2>&1",
            f"tmux kill-session -t {session}",
            f"tmux new-session -s {session} -n {windows[0]} -d {shell}",
    ]
    for w in windows[1:]:
        cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
    cmds += ["sleep 1"]

    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds, notes


def run(args):
    cmds, notes = create_commands("spiral", args)
    if args.dry_run:
        print("Dry-run mode due to -n flag, otherwise the following commands would be executed:")
    else:
        print("Executing the following commands:")

    print("\n".join(cmds))
    print("")

    if not args.dry_run:
        os.environ["TMUX"] = ""
        os.system("\n".join(cmds))

    print('\n'.join(notes))


if __name__ == "__main__":
    from config import get_args
    args = get_args()
    run(args)
