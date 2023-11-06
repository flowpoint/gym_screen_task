import gymnasium as gym
from gymnasium import spaces
import numpy as np

import string

class ScreenEnv(gym.Env):
    # start with button clicking in a gridworld
    # cursor is moved with a relative position
    metadata = {"render_modes": ["human", "array"]}

    def __init__(self, render_mode=None, resolution=[128,128]):
        self.resolution = resolution
        self.render_mode = render_mode

        max_text_context = 512

        # screen pixels in rgb
        self.observation_space = spaces.Dict(
                {
                    "screen": spaces.Box(
                        low=0,
                        high=2,
                        shape=np.array(resolution, dtype=np.int64),
                        dtype=int
                        ),
                    "task_description": spaces.Text(
                        min_length=0,
                        max_length=512,
                        charset=string.printable,
                        )
                    }
                )


        # maybe replace this with the easier space.Text actionspace

        num_keyboard_keys = len(string.printable)
        # treat press and release as separate actions
        num_keyboard_actions = num_keyboard_keys*2

        num_simultaneous_keys = 3
        self.keyboard_space = spaces.MultiDiscrete([num_keyboard_actions, num_simultaneous_keys])

        # mouse presses
        # left, right and scroll-wheel button
        num_mouse_buttons = 3
        # press and release
        num_mouse_actions = num_mouse_buttons * 2
        self.mouse_button_space = spaces.MultiDiscrete([num_mouse_actions, 2])

        # mouse movement (relative)
        max_mouse_velocity = 10. # in some strange metric like pixel/frame
        self.mouse_rel_move = spaces.Box(
                low=-max_mouse_velocity,
                high=max_mouse_velocity,
                shape=np.array([1,1], dtype=np.int64),
                dtype=float
                )

        # mouse movement (absolute)
        self.mouse_abs_move = spaces.Box(
                low=np.array([0,0], dtype=np.int64),
                high=np.array(self.resolution, dtype=np.int64),
                shape=np.array([2], dtype=np.int64),
                dtype=int
                )

        # mouse scroll
        max_mouse_scroll_velocity = 10. # ???
        self.mouse_scroll = spaces.Box(
                low=-max_mouse_scroll_velocity,
                high=max_mouse_scroll_velocity,
                shape=(1,),
                dtype=float
                )
        
        self.action_space = spaces.Dict(
                {
                "keyboard": self.keyboard_space,
                "mouse_buttons": self.mouse_button_space,
                "mouse_rel_move": self.mouse_rel_move,
                "mouse_abs_move": self.mouse_abs_move,
                "mouse_scroll": self.mouse_scroll,
                }
                )

        self.cursor_pos = np.array([0,0])
        self.cursor_shape = np.array([[1]])
        self.cursor_size = self.cursor_shape.shape

        self.button_pos = np.array([16,16])
        assert all(np.zeros([2]) <= self.button_pos)
        assert all(self.button_pos < self.resolution)
        self.button_shape = np.array([[2]])
        self.button_size = self.cursor_shape.shape

    def _get_info(self):
        return {}

    def _get_obs(self):
        screenbuf = np.zeros(self.resolution, dtype=np.int64)

        x,y = self.button_pos
        xr = slice(x, x+self.button_size[0])
        yr = slice(y, y+self.button_size[1])
        screenbuf[xr, yr] = self.button_shape

        x,y = self.cursor_pos
        xr = slice(x, x+self.cursor_size[0])
        yr = slice(y, y+self.cursor_size[1])
        screenbuf[xr, yr] = self.cursor_shape
        return {"screen":screenbuf, "task_description":""}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cursor_pos = [0,0]

        obs = self._get_obs()
        info = self._get_info()
        self.render()
        return obs, info

    def render(self):
        if self.render_mode == 'human':
            raise NotImplemented()
        elif self.render_mode == 'array':
            print(self._get_obs()['screen'])

    def step(self, action):
        raise NotImplemented()


    def close(self):
        pass

    def _get_step_reward(self, action):
        if all(self.cursor_pos == self.button_pos):
            reward = 1.
            terminated = True
        else: 
            reward = 0.
            terminated = False

        return reward, terminated

    def step(self, action):
        movement = [
                [0,1],
                [1,0],
                [0,-1],
                [-1,0],
                [0,0],
                ]

        self.cursor_pos = (np.array(self.cursor_pos) + movement[action]).clip(np.array([0,0]), self.resolution)
        reward, terminated = self._get_step_reward(action) 

        obs = self._get_obs()
        info = self._get_info()
        self.render()
        truncated = False
        return obs, reward, terminated, truncated, info



'''
class DragSlider(ScreenEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
'''


