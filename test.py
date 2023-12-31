import gym_screen_task
import gymnasium as gym
import gym_screen_task
import numpy as np

print(gym.pprint_registry())

#env = gym.make('gym_screen_task/click_button-v0')
#render_mode = None
render_mode = 'array'
#env = gym.make('gym_screen_task/click_button-v0', render_mode=render_mode, resolution=[32,32])
env = gym.make('gym_screen_task/click_button-v1', render_mode=render_mode, resolution=[32,32])
#env = gym.make('gym_screen_task/drag_slider-v0', render_mode=render_mode, resolution=[32,32])

np.set_printoptions(linewidth=128, edgeitems=128)
obs, info = env.reset()

for i in range(16):
    obs, reward, term, trunc, info = env.step({'mouse_rel_move':[1,1], 'mouse_buttons':[0,0,0]})
    print(reward)

# dont move
# press mouse button 1 [left] down and release/keep released the others
obs, reward, term, trunc, info = env.step({'mouse_rel_move':[0,0], 'mouse_buttons':[1,0,0]})
# dont move and release mouse button 1
obs, reward, term, trunc, info = env.step({'mouse_rel_move':[0,0], 'mouse_buttons':[0,0,0]})
print(reward)
assert reward == 1.


