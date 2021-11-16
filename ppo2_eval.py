#!/usr/bin/env python

"""
Evaluation PPO2 agent on local validation sets with MPI
"""

import tensorflow as tf
import datetime

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import baselines.ppo2.ppo2 as ppo2

import csv
import os
import sys
from functools import partial

from sonic_util import make_env

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver, RunObserver

timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')

ex = Experiment()

def create_eval_envs():
    env_fns = [] 
    env_names = []
    with open('sonic-validation.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if 'Sonic' in row[0]:
                print('Add Environments: ', row[0] + ': ' + row[1])
                env_fns.append(partial(make_env, game=row[0], state=row[1]))
                env_names.append(row[0] + '-' + row[1])

    return env_fns, env_names

@ex.main
def main(rank):
    """Run PPO until the environment throws an exception."""
    # Parallel Evaluation

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    env_fns, env_names = create_eval_envs()
    num_games = len(env_names)

    ex.observers.append(FileStorageObserver.create('./logs/ppo2_reptile_eval_{}_'.format(env_names[rank])+timestamp))
    print("rank = {}, env = {}".format(rank, env_names[rank]))

    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(network='cnn',
                   env=DummyVecEnv([env_fns[rank]]),
                   nsteps=8912,
                   nminibatches=8,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=3,
                   log_interval=1,
                   ent_coef=0.001, # lower entropy for fine-tuning
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   total_timesteps=int(1.05e6),
                   load_path='logs/reptile_4/checkpoints/00080', # Pretrained model
                   
        )

if __name__ == '__main__':
    rank = int(sys.argv[1])
    main(rank)
