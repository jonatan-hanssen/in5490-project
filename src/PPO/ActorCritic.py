import argparse
import os
from distutils.util import strtobool
import time

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def init_weightsNbias(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Descr:
        Most, if not all implementations of PPO seen so far use the approach
        of initializing weights orthogonaly. According to both the authors behind
        Proximal Policy Optimization, and the paper by title "Exact solutions to
        the nonlinear dynamics of learning in deep learning networks", the approach
        of creating such initial conditions leads to faithful propagation of gradients,
        and better convergence (See page 2 of the latter paper).
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(env_name, seed):
    def env_gen():
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return env_gen


class Agent(nn.Module):
    """
    Descr:
        The agents task is to decide the next action to perform, and evaluate the
        possible future rewards, a.k.a. the value function, based on the current state
        the agent is in. In our case the agents chooses the

    """

    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            init_weightsNbias(
                nn.Linear(
                    np.array(envs.single_observation_space["image"].shape).prod(), 64
                )
            ),
            nn.Tanh(),
            init_weightsNbias(nn.Linear(64, 64)),
            nn.Tanh(),
            init_weightsNbias(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            init_weightsNbias(
                nn.Linear(
                    np.array(envs.single_observation_space["image"].shape).prod(), 64
                )
            ),
            nn.Tanh(),
            init_weightsNbias(nn.Linear(64, 64)),
            nn.Tanh(),
            init_weightsNbias(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, X):
        return self.critic(X)

    def get_action_and_value(self, X, action=None):
        logits = self.actor(X)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(X)


if __name__ == "__main__":
    envs = gym.vector.SyncVectorEnv([make_env("MiniGrid-UnlockPickup-v0", 42)])
    # print(envs.observation_space)
    # print(envs.single_observation_space)
    # print(envs.observation_space["image"])
    # print(np.array((envs.single_observation_space["image"]).shape).prod())
    # print(envs.single_action_space.n)
    # observation, info = envs.reset(seed=42)
    # print(observation["image"])
    # print(observation.transpose(2,0,1))
    agent = Agent(envs)
    # print(agent.state_dict())
    # src_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    src_dir = os.path.dirname(__file__)

    torch.save(agent.state_dict(), src_dir + "/PPO.txt")

    model = Agent(envs)
    model.load_state_dict(torch.load(src_dir + "/PPO.txt"))
    
    # next_observation = torch.Tensor(envs.reset()[0])


    # action, _, _, _ = agent.get_action_and_value(next_observation)
    # print(action)
