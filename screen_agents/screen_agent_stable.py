import gym_screen_task
import gymnasium as gym
import gym_screen_task
import numpy as np

import io

#print(gym.pprint_registry())

import stable_baselines3
from stable_baselines3 import PPO

np.set_printoptions(linewidth=128, edgeitems=128)

class FilterAction(gym.ActionWrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self.action_space = env.action_space[key]

    def action(self, act):
        return {'mouse_rel_move': act}

class FilterActions(gym.ActionWrapper):
    def __init__(self, env, filter_keys):
        super().__init__(env)
        self.action_space = gym.spaces.Dict({key:env.action_space[key] for key in filter_keys})

    def action(self, act):
        return {'mouse_rel_move': act}

'''
class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.observation_space = None

    def observation(self, obs):
        return obs
'''

render_mode = ['array', 'human'][1]

def get_env(noise):
    env = gym.make(
            'gym_screen_task/click_button-v1', 
            render_mode=render_mode, 
            resolution=[64,64], 
            timelimit=100, 
            noise=noise, 
            frame_stack=1,
            random_cursor_start=True)
    env = gym.wrappers.FilterObservation(env, filter_keys=['screen'])
    #env = gym.wrappers.TimeLimit(env, max_episode_steps=580)
    def obst(d):
        d['screen'] = d['screen'] / 3.
        return d

    #env = gym.wrappers.TransformObservation(env, obst)
    #env = FilterAction(env, filter_keys=['mouse_rel_move', 'mouse_buttons'])
    env = FilterAction(env, key='mouse_rel_move')
    #env = gym.wrappers.FrameStack(env, 2)
    #env = gym.make('gym_screen_task/drag_slider-v0', render_mode=render_mode, resolution=[32,32])
    return env


#model = PPO('MlpPolicy', env, verbose=1)
lr = 0.0003
#lr = 0.1
n_cpu = 4
#env = stable_baselines3.common.vec_env.SubprocVecEnv([get_env(noise=63) for _ in range(n_cpu)], 'spawn')
#env = stable_baselines3.common.vec_env.DummyVecEnv([get_env(noise=63)])
env = get_env(noise=63)

#bs = 8*2048
bs = 64
model = PPO('MultiInputPolicy', env, verbose=1, learning_rate=lr, batch_size=bs)

num_epochs = 10



##
def train_w_render(learn=True, noise=6):
    for ep in range(num_epochs):
        #noise = 31 / 10 * ep #+ ep*0.03
        env = get_env(noise)
        model.set_env(env)

        vec_env = model.get_env()
        vec_env.reset()
        vec_env.eps = 10000


        if learn:
            model.learn(total_timesteps=10000, reset_num_timesteps=False)
        term = False
        obs = vec_env.reset()
        s = 0
        while not term:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, term, info = vec_env.step(action)
            s += 1
            vec_env.render()
            if not term and 0:
                print(action)
                print(info)
                print(s)
                #print(obs['screen'].reshape([4, 32,32]))
                #for o in obs['screen']:
                #print(o.reshape([32,32]))

        print(reward, s)

#train_w_render(learn=True)
model.learn(total_timesteps=2000_000)
