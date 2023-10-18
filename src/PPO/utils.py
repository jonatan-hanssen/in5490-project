import json
import gymnasium as gym
import random
import numpy as np
import torch


def make_env(env_name, seed, render_mode="rgb_array"):
    """
    Descr:
        Returns a "vectorized" environenment, meaning that it is wrapped in a gymnasium vector, allowing for
        some aditional functionality, such as parallel training of many environments.
    """

    def env_gen():
        env = gym.make(env_name, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return env_gen


def read_params(params):
    file = open(params)
    params = json.load(file)
    # print(json.dumps(params, indent=4, separators=(":", ",")))
    return params


def save_params(params):
    file = open("hyperparams.json", "w")
    json.dump(params, file, indent=4, separators=(",", ":"))


def seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
