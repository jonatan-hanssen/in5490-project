import gymnasium as gym
import matplotlib.pyplot as plt
import time
import constants
import numpy as np

from utils import obs_to_string, llama2_70b_policy

env = gym.make("MiniGrid-UnlockPickup-v0", render_mode="human")


# observation, info = env.reset(seed=1)
observation, info = env.reset()

policy = llama2_70b_policy()

action_list = list()

for _ in range(1000):
    if not action_list:
        prompt = obs_to_string(observation["image"])
        answer = policy(prompt)

        if "FORWARD" in answer:
            action_list.append(2)
        elif "LEFT" in answer:
            action_list.append(0)
            action_list.append(2)
        elif "RIGHT" in answer:
            action_list.append(1)
            action_list.append(2)

    action = action_list.pop(0)

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
