#!/usr/bin/env python

"""
Evaluation PPO2 agent on local validation sets with MPI
"""

import tensorflow as tf
import datetime

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import baselines.ppo2.ppo2 as ppo2

from mpi4py import MPI
import csv
import os
from functools import partial

from sonic_util import make_env

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver, RunObserver

timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')

ex = Experiment()
ex.observers.append(FileStorageObserver.create('./logs/ppo2_reptile_eval_'+timestamp))

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

@ex.automain
def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    # Use MPI for parallel evaluation
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    env_fns, env_names = create_eval_envs()
    num_games = len(env_names)
    
    # assert size > 1
    process_per_env = int(MPI.COMM_WORLD.Get_size() / len(env_names))
    env_fns = env_fns*process_per_env
    env_names = env_names*process_per_env
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"[rank % 1]

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
                   total_timesteps=int(1e7),
                   load_path='logs/reptile_4/checkpoints/00080', # Pretrained model
        )


