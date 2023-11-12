import gymnasium as gym

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
