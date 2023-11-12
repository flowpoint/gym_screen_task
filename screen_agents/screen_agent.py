import gymnasium as gym
import gym_screen_task
import numpy as np

import torch
import torch.nn as nn
import torchrl as rl
#from tensordict
device = 'cpu'
lr = 3e-4
max_grad_norm = 1.
sub_batch_size = 64
num_episodes = 10

clip_epsilon = (
    0.2
    )

gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

#env = gym.make('gym_screen_task/click_button-v0')
render_mode = 'array'
#env = gym.make('gym_screen_task/click_button-v0', render_mode=render_mode, resolution=[32,32])
g_env = gym.make('gym_screen_task/click_button-v1', render_mode=render_mode, resolution=[32,32])
#env = gym.make('gym_screen_task/drag_slider-v0', render_mode=render_mode, resolution=[32,32])
w_env = gym.wrappers.FilterObservation(g_env, filter_keys=['screen'])
env = rl.envs.libs.gym.GymWrapper(w_env)

#rollout = env.rollout(3)
#rollout = env.rand_step()
rollout = env.action_spec.rand()
rollout = env.action_spec.rand()
#print(rollout)


num_cells = 32*32
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

policy_module = rl.TensorDictModule(
        actor_net, in_keys=['screen'], out_keys=[],
        )


exit(0)
np.set_printoptions(linewidth=128, edgeitems=128)


for ep in range(num_episodes):
    obs, info = env.reset()
    while not term:
        obs, reward, term, trunc, info = env.step({'mouse_rel_move':[1,1], 'mouse_buttons':[0,0,0]})
        print(reward)


# dont move
# press mouse button 1 [left] down and release/keep released the others
obs, reward, term, trunc, info = env.step({'mouse_rel_move':[0,0], 'mouse_buttons':[1,0,0]})
# dont move and release mouse button 1
obs, reward, term, trunc, info = env.step({'mouse_rel_move':[0,0], 'mouse_buttons':[0,0,0]})
print(reward)
assert reward == 1.


