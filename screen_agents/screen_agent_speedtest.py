import gymnasium as gym
import gym_screen_task
import numpy as np

import cProfile, pstats, io
from pstats import SortKey

import torch
import torch.nn as nn
#from tensordict
device = 'cpu'
lr = 3e-4
max_grad_norm = 1.
sub_batch_size = 64
num_episodes = 100

clip_epsilon = (
    0.2
    )

gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

#env = gym.make('gym_screen_task/click_button-v0')
render_mode = 'array'
#env = gym.make('gym_screen_task/click_button-v0', render_mode=render_mode, resolution=[32,32])
env = gym.make('gym_screen_task/find_button-v0', render_mode=render_mode, resolution=[32,32])
#env = gym.make('gym_screen_task/drag_slider-v0', render_mode=render_mode, resolution=[32,32])
#env = gym.wrappers.FilterObservation(env, filter_keys=['screen'])
#env = rl.envs.libs.gym.GymWrapper(w_env)


from timeit import timeit

def run_env():
    steps = 0
    for ep in range(num_episodes):
        term = False
        obs, info = env.reset()
        while not term:
            #obs, reward, term, trunc, info = env.step({'mouse_rel_move':[1,1], 'mouse_buttons':[0,0,0]})
            obs, reward, term, trunc, info = env.step([0,0])
            #obs, reward, term, trunc, info = env.step(env.action_space.sample())
            steps += 1

    print(f'steps: {steps}')


ti = timeit(run_env, number=1)
print(f'time: {ti}')

with cProfile.Profile() as pr:
    run_env()
    pr.print_stats()


exit()
# dont move
# press mouse button 1 [left] down and release/keep released the others
#obs, reward, term, trunc, info = env.step({'mouse_rel_move':[0,0], 'mouse_buttons':[1,0,0]})
# dont move and release mouse button 1
#obs, reward, term, trunc, info = env.step({'mouse_rel_move':[0,0], 'mouse_buttons':[0,0,0]})
#print(reward)
#assert reward == 1.


