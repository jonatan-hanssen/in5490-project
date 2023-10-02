import torch
import torch.nn as nn
from gymnasium import spaces
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to torche number of unit for torche last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2,2)),
            nn.Tanh(),
            nn.Conv2d(16, 32, (2,2)),
            nn.Tanh(),
            # nn.Conv2d(32, 64, (2,2)),
            # nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.Tanh())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
# env = make_vec_env("MiniGrid-UnlockPickup-v0", n_envs=1)
env = DummyVecEnv([lambda: ImgObsWrapper(gym.make("MiniGrid-UnlockPickup-v0", render_mode="rgb_array"))])
env.seed(42)
# env = ImgObsWrapper(env)

model = PPO(
    "CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1
)
model.learn(total_timesteps=2e4)
model.save("ppo_minigrid")
del model



