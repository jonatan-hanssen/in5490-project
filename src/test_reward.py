import gymnasium as gym
import matplotlib.pyplot as plt
import time
import constants
import numpy as np

from utils import obs_to_string, llama2_7b_reward_shaper

env = gym.make("MiniGrid-UnlockPickup-v0", render_mode="human")


# observation, info = env.reset(seed=1)
observation, info = env.reset()

rewarder = llama2_7b_reward_shaper(observation["mission"])


for _ in range(1000):


    # stringy = [[f"{cell[0]}{cell[1]}{cell[2]}" for cell in row] for row in observation["image"]]
    # for row in stringy:
    #     print(row)
    obs = obs_to_string(observation["image"], False, False)
    action = 1

    rewarder.suggest(obs)

    action = constants.ACTION_TO_IDX[input("action: ")]


    rewarder.compare(action, observation["image"])
    observation, reward, terminated, truncated, info = env.step(action)


    if terminated or truncated:
        observation, info = env.reset()
env.close()
