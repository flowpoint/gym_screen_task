import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ScreenEnv(gym.Env):
    # start with button clicking in a gridworld
    # cursor is moved with a relative position
    metadata = {"render_modes": ["human", "array"]}

    def __init__(self, render_mode=None, resolution=[128,128]):
        self.resolution = resolution
        self.render_mode = render_mode

        self.observation_space = spaces.Dict(
                {
                    "screen": spaces.Box(
                        low=0,
                        high=1,
                        shape=np.array(resolution, dtype=np.int64),
                        dtype=int
                        )
                    }
                )


        # wasd and click
        self.action_space = spaces.Discrete(5)

        self.cursor_pos = np.array([0,0])
        self.button_pos = np.array([64,64])


    def _get_obs(self):
        screenbuf = np.zeros(self.resolution, dtype=np.int64)
        x,y = self.cursor_pos
        screenbuf[x][y] = 1
        return {"screen":screenbuf}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cursor_pos = [0,0]

        obs = self._get_obs()
        info = self._get_info()
        self.render()
        return obs, info
    
    def step(self, action):
        movement = [
                [0,1],
                [1,0],
                [0,-1],
                [-1,0],
                [0,0],
                ]

        self.cursor_pos = (np.array(self.cursor_pos) + movement[action]).clip(np.array([0,0]), self.resolution)
        if all(self.cursor_pos == self.button_pos):
            reward = 1.
            terminated = True
        else: 
            reward = 0.
            terminated = False

        obs = self._get_obs()
        info = self._get_info()
        self.render()
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            raise NotImplemented()
        elif self.render_mode == 'array':
            print(self._get_obs()['screen'])


    def close(self):
        pass


