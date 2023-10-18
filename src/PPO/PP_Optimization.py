import argparse
import os
from distutils.util import strtobool
import time
import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from ActorCritic import *
from utils import *
import json


class PPO:
    def __init__(self, param_file):
        self.args = read_params(param_file)
        seeding(self.args["seed"])

        seeds = [random.randint(0, 20000) for _ in range(self.args["num_envs"])]

        self.envs = gym.vector.SyncVectorEnv(
            [
                make_env(self.args["env_name"], seeds[i])
                for i in range(self.args["num_envs"])
            ]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args["lr"], eps=1e-8
        )
        self.single_obs_shape = int(
            np.array(self.envs.single_observation_space["image"].shape).prod()
        )
        # print(envs.single_observation_space)

        # The following tensors need not to be initialized to again
        self.observations = torch.zeros(
            (self.args["ep_steps"], self.args["num_envs"], self.single_obs_shape),
        ).to(self.device)
        self.actions = torch.zeros(
            (self.args["ep_steps"], self.args["num_envs"])
            + self.envs.single_action_space.shape
        ).to(self.device)
        # The ActorCritic Network outputs log probabilities
        self.logprobs = torch.zeros((self.args["ep_steps"], self.args["num_envs"])).to(
            self.device
        )
        self.rewards = torch.zeros_like(self.logprobs).to(self.device)
        self.dones = torch.zeros_like(self.logprobs).to(self.device)
        self.values = torch.zeros_like(self.logprobs).to(self.device)
        self.advantages = torch.zeros_like(self.logprobs).to(self.device)
        self.returns = torch.zeros_like(self.logprobs).to(self.device)

    def do_it(self):
        self.args["batch_size"] = int(self.args["num_envs"] * self.args["ep_steps"])
        self.args["minibatch_size"] = int(
            self.args["batch_size"] // self.args["num_minibatches"]
        )
        optimizer = optim.Adam(self.agent.parameters(), lr=self.args["lr"], eps=1e-5)
        # _, _ = envs.reset(seed=self.args["seed"])
        self.args["rollouts"] = int(self.args["tot_steps"] // self.args["batch_size"])

        # Initializing the next step
        next_observation = torch.Tensor(self.envs.reset()[0]["image"].flatten()).to(
            self.device
        )
        next_done = torch.zeros(self.args["num_envs"]).to(self.device)
        # next_done = torch.zeros(.args.num_envs).to(device)

        # Episode: Moving n steps and estimating the value funvtion for each step
        for rollout in range(self.args["rollouts"] + 1):
            if self.args["anneal_lr"]:
                frac = 1.0 - (rollout - 1.0) / self.args["rollouts"]
                lrnow = frac * self.args["lr"]
                self.optimizer.param_groups[0]["lr"] = lrnow

            print(f"Rollout num: {rollout}")
            next_done, next_observation = self.next_episode(next_done, next_observation)
            print(f"Sum of rewards per episode: {torch.sum(self.rewards)}")
            self.PPO_update()

    def next_episode(self, next_done, next_observation):
        """Steps through a single episode and calculates what is needed to
        perform PPO update
        """
        for step in range(self.args["ep_steps"]):
            # global_step += args["num_envs"]
            self.observations[step] = next_observation
            self.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    next_observation
                )
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            next_observation, reward, done, truncated, info = self.envs.step(
                np.array([action.cpu().numpy()])
            )
            # Converting to tensor and transfering to gpu for faster computation
            self.rewards[step] = torch.Tensor(reward).to(self.device).view(-1)

            # print(next_observation["image"])

            next_observation, next_done = (
                torch.Tensor(next_observation["image"].flatten()).to(self.device),
                torch.Tensor(next_done).to(self.device),
            )

        # Now that the agent(-s) has(-ve) played out an episode, it's time
        # to backtrack all steps, and compute the discounted rewards
        with torch.no_grad():
            next_value = self.agent.get_value(next_observation).reshape(1, -1)
            # General advanatage estimation proposed in the original paper on PPO
            if self.args["gae"]:
                # self.returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.args["ep_steps"])):
                    if t == self.args["ep_steps"] - 1:
                        # "nextnonterminal" is a environment specific variable indicating if the
                        # agent has finished the game before reaching time step limit
                        nextnonterminal = 1.0 - next_done
                        next_values = next_value
                        last_gae_lam = 0
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_values = self.values[t + 1]
                        last_gae_lam = self.returns[t + 1]

                    delta = (
                        self.rewards[t]
                        + self.args["gamma"] * next_values * nextnonterminal
                        - self.values[t]
                    )
                    self.advantages[t] = (
                        delta
                        + self.args["gamma"]
                        * self.args["gae_lambda"]
                        * nextnonterminal
                        * last_gae_lam
                    )
                self.returns = self.advantages + self.values

            else:
                # self.returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.args["ep_steps"])):
                    if t == self.args["ep_steps"] - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = self.returns[t + 1]

                    self.returns[t] = (
                        self.rewards[t]
                        + self.args["gamma"] * nextnonterminal * next_return
                    )
                self.advantages = self.returns - self.values

        return next_done, next_observation

    def minibatch_generator(self):
        """This generates a shuffled list of minibatches
        When next_episode is called, this will have new values because the self values will be changed
        """
        b_obs = self.observations.view((-1, self.single_obs_shape))
        b_logs = self.logprobs.view(
            -1,
        )
        b_acts = self.actions.view((-1,) + self.envs.single_action_space.shape)
        b_advs = self.advantages.view(
            -1,
        )
        b_vals = self.values.view(
            -1,
        )
        b_rets = self.returns.view(
            -1,
        )

        batch_idxs = np.random.choice(
            self.args["batch_size"], self.args["batch_size"], replace=False
        )
        for start in range(0, self.args["batch_size"], self.args["minibatch_size"]):
            end = start + self.args["minibatch_size"]
            minibatch_idxs = batch_idxs[start:end]

            yield (
                b_obs[minibatch_idxs],
                b_logs[minibatch_idxs],
                b_acts[minibatch_idxs],
                b_advs[minibatch_idxs],
                b_vals[minibatch_idxs],
                b_rets[minibatch_idxs],
            )

    def PPO_update(self):
        # Flattening all containers in order to compute components of surrogate losses
        # using minibatches
        batch_observations = self.observations.view((-1, self.single_obs_shape))
        batch_logprobs = self.logprobs.view(-1)
        batch_actions = self.actions.view((-1,) + self.envs.single_action_space.shape)
        batch_advantages = self.advantages.view(-1)
        batch_values = self.values.view(-1)
        batch_returns = self.returns.view(-1)

        # batch_idx = np.arange(args["batch_size"])
        for epoch in range(self.args["epochs"]):
            # np.random.shuffle(batch_idx)
            # NOTE this shuffles but in a greeeg way

            for values in self.minibatch_generator():
                (
                    batch_observations,
                    batch_logprobs,
                    batch_actions,
                    batch_advantages,
                    batch_values,
                    batch_returns,
                ) = values

                _, new_logprob, entropy, new_value = self.agent.get_action_and_value(
                    batch_observations, batch_actions.long()
                )
                # print(new_logprob.exp())

                # The probability ratio of the new policy vs the old policy
                # this is equivalent to prob / prob_old
                policy_ratio = (new_logprob - batch_logprobs).exp()

                p_grad_clip1 = policy_ratio * batch_advantages
                p_grad_clip2 = batch_advantages * torch.clamp(
                    policy_ratio,
                    1.0 - self.args["clip_epsilon"],
                    1.0 + self.args["clip_epsilon"],
                )
                # Whenever a paper writes an equation as the expected value, take the mean() of that function
                # to imitate the expected value
                clip_loss = torch.min(p_grad_clip1, p_grad_clip2).mean()

                # Computing Value Loss
                value_loss = (
                    (new_value.view(-1) - batch_returns) ** 2
                ).mean() * self.args["value_loss_coef"]

                # Computing expected value of Entropy Loss
                entropy_loss = entropy.mean() * self.args["entropy_coef"]

                surrogate_loss = clip_loss - value_loss + entropy_loss

                self.optimizer.zero_grad()
                surrogate_loss.backward()
                # nn.utils.clip_grad_norm_(self.agent.parameters(), self.args["max_grad_norm"])
                self.optimizer.step()


if __name__ == "__main__":
    ppo = PPO("hyperparams.json")
    # save_params(self.self.args)
    ppo.do_it()
