from gym_screen_task.wrappers import FilterAction
import gym_screen_task
import gymnasium as gym
import gym_screen_task
import numpy as np

from multiprocessing import freeze_support
import os

import io

#print(gym.pprint_registry())

import stable_baselines3
from stable_baselines3 import PPO,DQN,A2C
from stable_baselines3.common.callbacks import EvalCallback
from screen_agents.custom_extractor import CustomCombinedExtractor

np.set_printoptions(linewidth=128, edgeitems=128)

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
                random_cursor_start=True, 
                envid=rank)
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


def get_vec_env(n_cpu, hparams, dummy=False):
    print(f'using {n_cpu} parallel environments')
    #env = make_env(0)()
    if dummy == False:
        env = stable_baselines3.common.vec_env.SubprocVecEnv([make_env(rank, hparams) for rank in range(n_cpu)], 'spawn')
    else:
        env = stable_baselines3.common.vec_env.DummyVecEnv([make_env(rank, hparams) for rank in range(n_cpu)])

    env = stable_baselines3.common.vec_env.VecMonitor(env)
    #env = stable_baselines3.common.vec_env.VecFrameStack(env, 2)
    env = stable_baselines3.common.vec_env.VecTransposeImage(env)
    return env

# apparantly working params:
# lr: 0.0003
# 'bs': 64,#32*16*4*8*2048,
# gamma: 0.5
# clip: 0.2
# learned after around 8M timesteps to a best reward of -6 and ~60 episode steps
# resolution 64,64
# fully random cursor and button pos
res = [32,32]
hparams = {
        'lr': 0.0003,
        'bs': 64,#8*2048,
        'resolution': res,
        'noise': max(res)-1,
        #'noise': 4,
        'gamma':0.99,
        'clip':0.2,
        'agent': 'ppo'
        }

def get_model(env, hparams):
    policy_kwargs = dict(
        #net_arch=dict(pi=[8, 8192], vf=[32, 32]),
        net_arch=dict(pi=[8,8,4], vf=[8,8,4]),
        features_extractor_class=CustomCombinedExtractor,
        #features_extractor_kwargs=dict(features_dim=128),
    )

    agent = hparams['agent']

    if agent == 'ppo':
        model = PPO('MultiInputPolicy', env, 
                    verbose=1, learning_rate=hparams['lr'], 
                    batch_size=hparams['bs'], gamma=hparams['gamma'],
                    policy_kwargs=policy_kwargs,
                    clip_range=hparams['clip'],
                    device='cpu')
    elif agent == 'a2c':
        model = A2C('MultiInputPolicy', env, policy_kwargs=policy_kwargs)
    elif agent == 'dqn':
        raise NotImplemented('discrete mouse movement is not impl')
        model = DQN('MultiInputPolicy', env, policy_kwargs=policy_kwargs)
    else:
        raise RuntimeError('unknown agent specified')

    return model

def demo(checkpoint='longtrain1'):
    #n_cpu = len(os.sched_getaffinity(0))
    n_cpu = 1#len(os.sched_getaffinity(0))
    env = get_vec_env(n_cpu, hparams, dummy=True)
    model = get_model(env, hparams)
    if checkpoint is not None:
        #model = #model.load(checkpoint)
        model = PPO.load(checkpoint, env)

    num_epochs = 20
    for ep in range(num_epochs):
        vec_env = model.get_env()
        vec_env.reset()
        vec_env.eps = 10000

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


def main():
    #bs = 8*2048
    n_cpu = len(os.sched_getaffinity(0))
    env = get_vec_env(n_cpu, hparams)
    model = get_model(env, hparams)

    eval_callback = EvalCallback(
            env, 
            eval_freq=10000,
            deterministic=True, 
            render=False)

    callback = eval_callback 
    #train_w_render(learn=True)
    model.learn(total_timesteps=1_000_000, callback=eval_callback)
    model.save('longtrain1')
    #model.learn(total_timesteps=10_000_000, callback=eval_callback)
    #print(model.policy)




if __name__ == '__main__':
    freeze_support()
    #main()
    demo()

