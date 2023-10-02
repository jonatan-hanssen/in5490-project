import torch
import torch.nn as nn 
from gymnasium import spaces
import gymnasium as gym 

from stable_baselines3 import PPO 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv 

env = DummyVecEnv([lambda: ImgObsWrapper(gym.make("MiniGrid-UnlockPickup-v0", render_mode="human"))])
env.seed(42)
# env = ImgObsWrapper(env)

model = PPO.load("ppo_minigrid", env=env)

obs = env.reset() 
for _ in range(1000): 
    action = model.predict(obs)
    obs, reward, dones, info = env.step(action)

env.close()
