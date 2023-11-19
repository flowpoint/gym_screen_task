from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import numpy as np

class AimVecMonitor(VecMonitor):
    def __init__(self, venv, aim_run, *args, **kwargs):
        super().__init__(venv, *args, **kwargs)
        self.aim_run = aim_run
        self.episode_returns_aim = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths_aim = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        self.episode_returns_aim = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths_aim = np.zeros(self.num_envs, dtype=np.int32)

        return super().reset()

    def step_wait(self):
        obs, rewards, dones, new_infos = super().step_wait()

        self.episode_returns_aim += rewards
        self.episode_lengths_aim += 1

        for i in range(len(dones)):
            if dones[i]:
                self.aim_run.track(self.episode_returns_aim[i], name='ep_returns')
                self.aim_run.track(self.episode_lengths_aim[i], name='ep_lengths')

                self.episode_returns_aim[i] = 0
                self.episode_lengths_aim[i] = 0

        return obs, rewards, dones, new_infos
