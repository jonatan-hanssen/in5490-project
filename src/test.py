import gymnasium as gym
import matplotlib.pyplot as plt
import time
import constants
import numpy as np
import torch
from utils import caption_action


env = gym.make("MiniGrid-UnlockPickup-v0", render_mode="human")


obs, dog = env.reset()
print(obs)
obs = obs["image"]
observation = env.step(1)[0]["image"]
stringy = [[f"{cell[0]}{cell[1]}{cell[2]}" for cell in row] for row in observation]
for row in stringy:
    print(row)
print(observation.shape)
observation = torch.Tensor(observation).flatten()
print(observation.shape)

observation = np.array(observation.reshape((7, 7, 3)).to(torch.int64))
print(observation.shape)

stringy = [[f"{cell[0]}{cell[1]}{cell[2]}" for cell in row] for row in observation]
for row in stringy:
    print(row)

print(caption_action(1, observation))
env.close()
