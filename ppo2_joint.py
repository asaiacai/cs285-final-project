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
import baselines.ppo2.ppo2 as ppo2
import baselines.common as policies

from sonic_util import make_env

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver, RunObserver

timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')

ex = Experiment()
ex.observers.append(FileStorageObserver.create('./logs/ppo2_joint_lr2e-3'+timestamp))

def create_envs():
    env_fns = [] 
    env_names = []
    with open('sonic-train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if 'Sonic' in row[0]:
                print('Add Environments: ', row[0] + ': ' + row[1])
                env_fns.append(partial(make_env, game=row[0], state=row[1]))
                env_names.append(row[1])

    return env_fns, env_names


@ex.automain
def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    env_fns, env_names = create_envs()
    num_parallel = 1
    tasks = [ShmemVecEnv([ef] * num_parallel) for ef in env_fns]
    decay_steps = 1000
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        2e-3, decay_steps)
    ppo2.learn(network='cnn',
                env=tasks[0],
                tasks=tasks,
                task_names=env_names,
                nsteps=4600, 
                nminibatches=8, 
                lam=0.95,
                gamma=0.99,
                noptepochs=3, 
                log_interval=1, 
                ent_coef=0.01,
                lr=lambda _: 2e-4,
                maml_beta=lr_decayed_fn,
                task_batch=28,
                cliprange=lambda _: 0.1, 
                total_timesteps=int(1e9),
                save_interval=10,
                load_path='/tmp/openai-2021-11-24-15-42-09-488475/checkpoints/00290')


