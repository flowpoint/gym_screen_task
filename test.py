import gym_screen_task
import gymnasium as gym
import gym_screen_task
import numpy as np

print(gym.pprint_registry())

#env = gym.make('gym_screen_task/click_button-v0')
#render_mode = None
render_mode = 'array'
env = gym.make('gym_screen_task/click_button-v0', render_mode=render_mode, resolution=[32,32])

np.set_printoptions(linewidth=128, edgeitems=128)
obs, info = env.reset()

for i in range(10):
    obs, reward, term, trunc, info = env.step(4)
    print(reward)
