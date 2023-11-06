from gym_screen_task.env import ScreenEnv
from gymnasium.envs.registration import register

register(
        id='gym_screen_task/click_button-v0',
        entry_point='gym_screen_task:ScreenEnv',
        max_episode_steps=300,
        )
