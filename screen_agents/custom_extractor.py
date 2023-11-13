import torch
import torch.nn as nn
from gymnasium import spaces
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        '''

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        '''


        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "screen":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                torch.set_float32_matmul_precision('high')

                model = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(subspace.shape[0]*subspace.shape[1]*subspace.shape[2], 16),
                        nn.Dropout(),
                        nn.LeakyReLU(),
                        nn.Linear(16,16),
                        nn.Dropout(),
                        nn.LeakyReLU(),
                        nn.Linear(16,8),
                        )

                n_input_channels = 1
                pad = 'same'
                '''
                model = nn.Sequential(
                        nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=1, padding=pad),
                        nn.Dropout(),
                        nn.LeakyReLU(),
                        nn.Conv2d(2, 8, kernel_size=5, stride=1, padding=pad),
                        nn.Dropout(),
                        nn.LeakyReLU(),
                        nn.AvgPool2d(16),
                        nn.Flatten(),
                        nn.Linear(8*4,8),
                        nn.Dropout(),
                        nn.LeakyReLU(),
                        nn.Linear(8,8),
                        )
                '''
                extractors[key] = torch.compile(model)
                total_concat_size += 8
                #total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
                #total_concat_size += 2*subspace.shape[1] * subspace.shape[2]

            '''
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16
            '''

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
