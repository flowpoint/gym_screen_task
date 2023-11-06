from gym_screen_task.click_button_env import ClickButtonEnv as ClickButtonEnvV0
from gym_screen_task.screen_env import ScreenEnv, DragSliderEnv, ClickButtonEnv
from gymnasium.envs.registration import register

'''
register(
        id='gym_screen_task/screen_env-v0',
        entry_point='gym_screen_task:ScreenEnv',
        max_episode_steps=600,
        )
'''
register(
        id='gym_screen_task/click_button-v1',
        entry_point='gym_screen_task:ClickButtonEnv',
        max_episode_steps=600,
        )

register(
        id='gym_screen_task/drag_slider-v0',
        entry_point='gym_screen_task:DragSliderEnv',
        max_episode_steps=600,
        )

register(
        id='gym_screen_task/click_button-v0',
        entry_point='gym_screen_task:ClickButtonEnv',
        max_episode_steps=300,
        )
