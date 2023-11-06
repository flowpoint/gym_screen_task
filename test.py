import gym_screen_task
import gymnasium as gym
import gym_screen_task

print(gym.pprint_registry())

#env = gym.make('gym_screen_task/click_button-v0')
env = gym.make('gym_screen_task/click_button-v0')

obs, info = env.reset()
print(obs)
