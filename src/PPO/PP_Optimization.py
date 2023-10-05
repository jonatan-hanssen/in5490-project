import argparse
import os
from distutils.util import strtobool
import time
from torch.utils.tensorboard import SummaryWriter
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
                    np.array(
                        np.array(envs.single_observation_space["image"].shape)
                    ).prod(),
                    64,
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
                    np.array(
                        np.array(envs.single_observation_space["image"].shape)
                    ).prod(),
                    64,
                )
            ),
            nn.Tanh(),
            init_weightsNbias(nn.Linear(64, 64)),
            nn.Tanh(),
            init_weightsNbias(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

        # Yet to be integrated -> shall serve as the second actor
        # self.LM_actor = llama2_7b_policy()

    def get_value(self, X):
        return self.critic(X)

    def get_action_and_value(self, X, action=None):
        logits = self.actor(X)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(X)


class PPO:
    def __init__(self, args):
        self.args = args()
        self.seeding()

        seeds = [random.randint(0, 20000) for _ in range(env_n)]
        self.envs = gym.vector.SyncVector(
            [make_env(env_name, seeds[i]) for i in range(self.args.num_envs)]
        )
        self.agent = Agent(envs).to(device)
        self.optimizer = optim.Adam(agent.parameters(), lr=self.args.lr, eps=1e-5)
        _, _ = envs.reset(seed=self.args.seed)
        self.rollouts = self.args.tot_steps // self.args.batch_size

    def forward(self):
        # Containers for values needed in calculation of surrogate loss
        single_obs_shape = (
            int(np.array(envs.single_observation_space["image"].shape).prod()),
        )

        self.observations = torch.zeros(
            (self.args.ep_steps, self.args.num_envs, single_obs_shape),
        ).to(device)
        self.actions = torch.zeros(
            (self.args.ep_steps, self.args.num_envs) + self.envs.single_action_space
        ).to(device)
        self.logprobs = torch.zeros((self.args.ep_steps, self.args.num_envs)).to(device)
        self.rewards = torch.zeros((self.args.ep_steps, self.args.num_envs)).to(device)
        self.dones = torch.zeros((self.args.ep_steps, self.args.num_envs)).to(device)
        self.values = torch.zeros((self.args.ep_steps, self.args.num_envs)).to(device)

        # Initializing the next step
        next_observation = torch.zeros(self.envs.reset()[0]).to(device)
        next_done = torch.zeros(self.args.num_envs).to(device)

        # Episode: Moving n steps and estimating the value funvtion for each step
        for rollout in range(self.rollouts + 1):
            for step in self.args.ep_steps:
                self.global_step += self.args.num_envs
                self.observations[step] = next_observation
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent(next_observation)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                next_observation, reward, done, truncated, info = envs.step(
                    action.cpu().numpy()
                )
                # Converting to tensor and transfering to gpu for faster computation
                self.rewards[step] = torch.Tensor(reward).to(device).view(-1)

                next_observation, next_done = (
                    torch.Tensor(next_observation).to(device),
                    torch.Tensor(next_done).to(device),
                )

            # Now that the agent(-s) has(-ve) played out an episode, it's time
            # to backtrack all steps, and compute the discounted rewards
            with torch.no_grad():
                next_value = agent.get_value(next_observation).reshape(1, -1)
                # General advanatage estimation proposed in the original paper on PPO
                if self.args.gae:
                    returns = torch.zeros_like(rewards).to(device)
                    # lastgaelam = 0
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            # "nextnonterminal" is a environment specific variable indicating if the
                            # agent has finished the game before reaching time step limit
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]

                        delta = (
                            rewards[t]
                            + self.args.gamma * values[t + 1] * nextnonterminal
                            - values[t]
                        )
                        self.returns[t] = (
                            delta
                            + self.args.gamma
                            * self.args.gae_lambda
                            * nextnonterminal
                            * self.returns[t + 1]
                        )
                    self.advantages = self.returns + self.values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]

                        returns = (
                            self.rewards[t]
                            * self.args.gamma
                            * nextnonterminal
                            * returns[t + 1]
                        )
                    self.advantages = returns - self.values

    def seeding(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print("The Main function")
