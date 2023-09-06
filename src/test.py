import gymnasium as gym
import matplotlib.pyplot as plt
import time
import constants
import numpy as np

from utils import obs_to_string, llama2_70b_policy

env = gym.make("MiniGrid-UnlockPickup-v0", render_mode="human")


# observation, info = env.reset(seed=1)
observation, info = env.reset()

policy = llama2_70b_policy(temperature=0.1)

action_list = list()

for _ in range(1000):
    if not action_list:
        prompt = obs_to_string(observation["image"])
        print(prompt)
        policy(prompt, action_list, env)

    print([constants.IDX_TO_ACTION[idx] for idx in action_list])
    action = action_list.pop(0)

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
