import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import string

class ScreenEnv(gym.Env):
    # start with button clicking in a gridworld
    # cursor is moved with a relative position
    metadata = {"render_modes": ["human", "array"], "render_fps": 60}

    def __init__(self, render_mode=None, resolution=[128,128], noise=None, timelimit=600, frame_stack=1, random_cursor_start=False):
        self.resolution = np.array(resolution, dtype=np.int64)
        self.render_mode = render_mode
        self.random_cursor_start = random_cursor_start

        self.frame_stack = frame_stack

        self.window = None
        self.clock = None
        self.timelimit = timelimit-1

        max_text_context = 512
        self.fixed_noise = noise

        # screen pixels in rgb
        # but grayscale first
        self.num_channels = 1

        self.frameshape = resolution + [self.num_channels]
        self.observation_space = spaces.Dict(
                {
                    "screen": spaces.Box(
                        low=0,
                        high=255,
                        #shape=np.array([frame_stack] + list(self.frameshape), dtype=np.uint8),
                        shape=np.array(list(self.frameshape), dtype=np.uint8),
                        #shape=np.array([1,31,31], dtype=np.uint8),
                        dtype=np.uint8
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
        max_mouse_velocity = 2. # in some strange metric like pixel/frame
        self.mouse_rel_move = spaces.Box(
                low=-max_mouse_velocity,
                high=max_mouse_velocity,
                shape=[2],
                dtype=float
                )

        # mouse movement (absolute)
        # mouse_abs_movement is ignored for now (notimplemented)
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

        # use a semantic image, with these classes
        # rgb graphics can be overlayed/generated on top of these
        self.semantic_class = {
                'cursor':128,
                'button':255,
                }

        # simple one pixel
        self.cursor_shape = np.ones([1,1]) * self.semantic_class['cursor']
        self.cursor_size = self.cursor_shape.shape

        # simple one pixel
        #self.button_shape = np.ones([1,1]) * self.semantic_class['button']
        #print(self.button_shape)
        #self.button_size = self.button_shape.shape

        self.eps = 0
        self.reset()
        self.bsc = 255


    def _get_info(self):
        return {'cursor': self.cursor_pos, "button": self.button_pos, "timestep": self.timestep}

    def _get_env_state(self):
        screenbuf = np.zeros(self.frameshape, dtype=np.uint8)
        x,y = self.button_pos
        '''
        xr = slice(x, x+self.button_size[0]+1)
        yr = slice(y, y+self.button_size[1]+1)
        screenbuf[xr, yr] = self.button_shape
        assert list(screenbuf.shape) == [64,64,1]
        assert int(self.semantic_class['button']) == 256
        '''
        screenbuf[x,y] = int(self.semantic_class['button'])
        assert np.any(screenbuf != 0.)

        #return {"screen":screenbuf, "task_description":""}
        return {"screen":screenbuf}


    def _get_frame(self):
        screenbuf = np.zeros(self.frameshape, dtype=np.uint8)
        x,y = self.button_pos
        screenbuf[x,y] = int(self.semantic_class['button'])
        assert np.any(screenbuf != 0.)

        # cursor overlays everything
        x2,y2 = round(self.cursor_pos[0]), round(self.cursor_pos[1])
        '''
        xr = slice(x, x+self.cursor_size[0]+1)
        yr = slice(y, y+self.cursor_size[1]+1)
        screenbuf[xr, yr] = self.cursor_shape
        '''
        screenbuf[x2,y2] = int(self.semantic_class['cursor'])
        frame = screenbuf
        return frame

    def _sample_new_button_pos(self):
        #self.button_pos = ((self.resolution-1)/2 + ((self.resolution-1)/2 * np.random.random([2]) * 0.16)).round().astype(np.int64)
        #self.button_pos = ((self.resolution-1)/2 + ((self.resolution-1)/2 * np.random.random([2]) * 0.16)).round().astype(np.int64) + ((np.random.random([2])-0.5)*4).round().astype(np.int64)
        bp = ((self.resolution-1)/2).round().astype(np.int64) + \
            ((np.random.random([2])-0.5)*self.noise).round().astype(np.int64)
        
        return bp

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.eps += 1
        if self.fixed_noise:
            self.noise = self.fixed_noise
        else:
            self.noise = 16 / 50000 * self.eps
        '''
        if self.eps % 500 == 0:
            pass
            #print(self.noise, self.eps)
        '''

        if self.random_cursor_start == True:
            self.cursor_pos = np.random.random([2]) * (self.resolution-1)
        else:
            self.cursor_pos = np.array(self.resolution)/2

        self.button_pos = self._sample_new_button_pos()
        assert all(np.zeros([2]) <= self.button_pos)
        assert all(self.button_pos < self.resolution)
        self.env_hidden_state = {}

        obs = self._get_frame()
        self.timestep = 0
        info = self._get_info()

        self.framehist = [np.zeros(self.frameshape, dtype=np.int64) for _ in range(self.frame_stack-1)] + [obs]

        framestack = np.stack(self.framehist, dtype=np.uint8)[0]

        #obs = {"screen":framestack, "task_description":""}
        obs = {"screen":framestack}

        return obs, info

    def _render_frame(self):
        self.window_size = 512

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / min(self.resolution[0], self.resolution[1])
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.button_pos,
                (pix_square_size, pix_square_size)
            )
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.cursor_pos + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        if self.render_mode == 'human':
            self._render_frame()
                    
        elif self.render_mode == 'array':
            print(self._get_frame()['screen'])
            pass

    def step(self, action):
        raise NotImplemented()


    def close(self):
        if self.window is not None:
                pygame.display.quit()
                pygame.quit()


    def _get_step_reward(self, action):
        raise NotImplemented('use a subclass instead of ScreenEnv directly')

    def _cursor_move(self, action):
        # ignore absolute_mouse movement for now (notimplemented)
        movement = action #action['mouse_rel_move']
        self.cursor_pos = (self.cursor_pos + movement).clip(np.array([0,0]), self.resolution-1)

    def _environ_step(self, action):
        self._cursor_move(action)

    def step(self, action):
        old_env_state = self._get_env_state()
        self._environ_step(action)
        new_env_state = self._get_env_state()
        
        self.timestep += 1
        reward, terminated = self._get_step_reward(action, old_env_state, new_env_state)

        new_obs = self._get_frame()

        info = self._get_info()

        if self.timestep >= self.timelimit:
            truncated = True
        else: 
            truncated = False

        #self.framehist.pop(0)
        #self.framehist.append(new_obs)
        #framestack = np.stack(self.framehist, dtype=np.uint8)[0]

        #assert list(framestack.shape) == [self.frame_stack, self.resolution[0], self.resolution[1], self.num_channels]

        #obs = {"screen":new_obs, "task_description":""}
        obs = {"screen": new_obs}

        return obs, reward, terminated, truncated, info


class FindButtonEnv(ScreenEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = self.mouse_rel_move
        self.observation_space = spaces.Dict({
            "screen": self.observation_space['screen']
            })

    def mouse_on_button(self, new_obs):
        cur = self.cursor_pos.round().clip(np.array([0,0]), self.resolution-1).astype(np.int64)
        assert list(cur.shape) == [2]
        return int(new_obs['screen'][cur[0], cur[1]]) == int(self.semantic_class['button'])

    def _get_step_reward(self, action, old_obs, new_obs):
        # use the convention of normalizing reward between [-1, 1] per step
        # define reward mixture

        # reward factor for finishing correct
        term_scale_factor = 1.
        # reward factor for minimizing distance
        dist_scale_factor = 0.5

        if self.mouse_on_button(new_obs):
            #print(f'mouse: {self.cursor_pos} button: {self.button_pos}')
            reward = 1. * term_scale_factor
        elif self.timestep >= self.timelimit:
            reward = -1.0 * term_scale_factor
        else: 
            cursor_button_dist = np.sqrt(((self.button_pos - self.cursor_pos)**2).sum()) 
            norm_dist = cursor_button_dist / np.sqrt((self.resolution**2).sum())
            assert 0. <= norm_dist <= 1.
            reward = dist_scale_factor * (0. - norm_dist)

            #reward = -0.0

        if self.timestep >= self.timelimit and self.mouse_on_button(new_obs):
            terminated = True
        else:
            terminated = False

        # normalize across factors again
        reward /= sum([term_scale_factor, dist_scale_factor])
        '''
        if self.mouse_on_button(new_obs) and action['mouse_buttons'][0] == 1:
            self.env_hidden_state['last_press_correct'] = True
            reward = 0.
            terminated = False
        elif self.mouse_on_button(new_obs) and action['mouse_buttons'][0] == 0:
            reward = 1.
            terminated = True
        else: 
            reward = 0.
            terminated = False
            self.env_hidden_state['last_press_correct'] = False
        '''

        return reward, terminated



class DragSliderEnv(ScreenEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mouse_on_button(self, new_obs):
        return new_obs['screen'][self.cursor_pos[0], self.cursor_pos[1]] == self.semantic_class['button']

    def _get_step_reward(self, action, old_obs, new_obs):
        if self.mouse_on_button(new_obs) and action['mouse_buttons'][0] == 1:
            self.env_hidden_state['slider_pressed'] = True
            reward = 0.
            terminated = False
        # slider slid 5 units to right and mouse was released
        elif self.cursor_pos[0] > self.button_pos[0]+5 and action['mouse_buttons'][0] == 0:
            reward = 1.
            terminated = True
        else: 
            reward = 0.
            terminated = False
            self.env_hidden_state['last_press_correct'] = False

        return reward, terminated

class PressKeyEnv(ScreenEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_step_reward(self, action, old_obs, new_obs):
        if action['keyboard'][0] == 1:
            self.env_hidden_state['last_press_correct'] = True
            reward = 0.
            terminated = False
        # slider slid 5 units to right and mouse was released
        elif action['keyboard'][0] == 0 and self.env_hidden_state['last_press_correct'] == True:
            reward = 1.
            terminated = True
        else: 
            reward = 0.
            terminated = False
            self.env_hidden_state['last_press_correct'] = False

        return reward, terminated




