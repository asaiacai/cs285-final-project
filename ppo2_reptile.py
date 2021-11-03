#!/usr/bin/env python

"""
Joint Train PPO on 47 training levels drawn from Sonic games
"""

import os
import datetime
import csv
from functools import partial

import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
import baselines.ppo2.ppo2_reptile as ppo2
from baselines import logger

from sonic_util import make_env

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver, RunObserver

timestamp = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')

ex = Experiment()
savedir = 'logs/ppo2_reptile_'+timestamp
ex.observers.append(FileStorageObserver.create(savedir))

def create_envs():
    env_fns = [] 
    env_names = []
    with open('sonic-train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if 'Sonic' in row[0]:
                print('Add Environments: ', row[0] + ': ' + row[1])
                env_fns.append(partial(make_env, game=row[0], state=row[1]))
                env_names.append(row[0] + '-' + row[1])

    return env_fns, env_names


@ex.automain
def main():
    """Run PPO until the environment throws an exception."""
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    env_fns, env_names = create_envs()
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(network='cnn',
                   env=ShmemVecEnv(env_fns]),
                   nsteps=8192, 
                   nminibatches=8, 
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=4, 
                   log_interval=1, 
                   ent_coef=0.01,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   eps_reptile=0.1,
                   total_timesteps=int(1e9),
                   save_interval=10,
                   load_path=None,)


