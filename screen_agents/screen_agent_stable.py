from gym_screen_task.wrappers import FilterAction
import gym_screen_task
import gymnasium as gym
import gym_screen_task
import numpy as np

import os

import io

#print(gym.pprint_registry())

import stable_baselines3
from stable_baselines3 import PPO


def make_env(rank, hparams):
    # inspired from https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-unleashing-the-power-of-vectorized-environments
    render_mode = ['array', 'human'][1]

    def _init():
        env = gym.make(
                'gym_screen_task/find_button-v0', 
                render_mode=render_mode, 
                resolution=hparams['resolution'], 
                timelimit=100, 
                noise=hparams['noise'], 
                frame_stack=1,
                random_cursor_start=True)
        #env = gym.wrappers.FilterObservation(env, filter_keys=['screen'])
        #env = gym.wrappers.TimeLimit(env, max_episode_steps=580)
        #env = gym.wrappers.TransformObservation(env, obst)
        #env = FilterAction(env, filter_keys=['mouse_rel_move', 'mouse_buttons'])
        #env = FilterAction(env, key='mouse_rel_move')
        #env = gym.wrappers.FrameStack(env, 2)
        #env = gym.make('gym_screen_task/drag_slider-v0', render_mode=render_mode, resolution=[32,32])
        env.reset()
        return env
    return _init


def main():
    np.set_printoptions(linewidth=128, edgeitems=128)

    #bs = 8*2048
    res = [64,64]
    hparams = {
            'lr': 0.0003,
            'bs': 64,
            'resolution': res,
            'noise': max(res)-1,
            }

    n_cpu = len(os.sched_getaffinity(0))
    print(f'using {n_cpu} parallel environments')
    #env = make_env(0)()
    env = stable_baselines3.common.vec_env.SubprocVecEnv([make_env(rank, hparams) for rank in range(n_cpu)], 'spawn')
    env = stable_baselines3.common.vec_env.VecMonitor(env)

    model = PPO('MultiInputPolicy', env, verbose=1, learning_rate=hparams['lr'], batch_size=hparams['bs'])

    num_epochs = 10

    def train_w_render(learn=True, noise=6):
        for ep in range(num_epochs):
            #noise = 31 / 10 * ep #+ ep*0.03
            #env = make_env(0, hparams)()
            #model.set_env(env)
            vec_env = model.get_env()
            vec_env.reset()
            vec_env.eps = 10000

            if learn:
                model.learn(total_timesteps=10000, reset_num_timesteps=False)
            term = [False]*n_cpu
            obs = vec_env.reset()
            s = 0
            while not all(term):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, term, info = vec_env.step(action)
                s += 1
                vec_env.render()
                '''
                if not term and 0:
                    print(action)
                    print(info)
                    print(s)
                    #print(obs['screen'].reshape([4, 32,32]))
                    #for o in obs['screen']:
                    #print(o.reshape([32,32]))
                '''

            #print(reward, s)
            print([inf['successful'] for inf in info])

    train_w_render(learn=True)
    #model.learn(total_timesteps=2000_000)



from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    main()

