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
    def __init__(self, param_file, llama_policy=False, llama_reward=False):
        self.args = read_params(param_file)
        seeding(self.args["seed"])

        self.env = gym.make(
            self.args["env_name"], render_mode="rgb_array", max_steps=self.args["steps"]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Agent(self.env, llama_policy).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args["lr"], eps=1e-8
        )
        self.obs_shape = int(np.array(self.env.observation_space["image"].shape).prod())

        self.reward_shaper = (
            llama2_reward_shaper(self.env.reset()[0]["mission"])
            if llama_reward
            else None
        )

        # Initialize all observation-, return-, valuevectors and so on
        self.reset_memory()

    def train(self):
        """Generates samples from a single episode, backpropagates the samples and
        repeats for rollout number of times
        """

        self.args["batch_size"] = self.args["steps"]
        self.args["minibatch_size"] = int(
            self.args["batch_size"] // self.args["num_minibatches"]
        )
        optimizer = optim.Adam(self.agent.parameters(), lr=self.args["lr"], eps=1e-5)

        # Initializing the next step

        # Episode: Moving n steps and estimating the value funvtion for each step
        for rollout in range(self.args["rollouts"]):
            print(f"Rollout num: {rollout}")
            self.next_episode()  # Perform steps and store relevant values
            print(f"Sum of rewards per episode: {torch.sum(self.rewards)}")
            self.PPO_update()  # Use values stored to backpropagate using PPO
            self.reset_memory()  # Restore
        save_model(self.agent)

    def next_episode(self):
        """Steps through a single episode and calculates what is needed to
        perform PPO update
        """
        observation_dict, _ = self.env.reset()
        observation = torch.Tensor(observation_dict["image"].flatten()).to(self.device)

        for step in range(self.args["steps"]):
            self.observations[step] = observation

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(observation)

            self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # LLM reward shaping
            if self.reward_shaper:
                # this sets self.suggestions
                self.reward_shaper.suggest(observation_dict["image"])
                advisor_reward = self.reward_shaper.compare(
                    action.cpu().numpy(), observation_dict["image"]
                )
            else:
                advisor_reward = 0

            observation_dict, reward, done, truncated, info = self.env.step(
                action.cpu().numpy()
            )

            self.rewards[step] = reward + advisor_reward

            observation = torch.Tensor(observation_dict["image"].flatten()).to(
                self.device
            )

            if done:
                break

        # Now that the agent(-s) has(-ve) played out an episode, it's time
        # to backtrack all steps, and compute the discounted rewards
        with torch.no_grad():
            next_value = self.agent.get_value(observation).reshape(1, -1)
            # General advanatage estimation proposed in the original paper on PPO
            if self.args["gae"]:
                # self.returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.args["steps"])):
                    if t == self.args["steps"] - 1:
                        # "nextnonterminal" is a environment specific variable indicating if the
                        # agent has finished the game before reaching time step limit
                        # nextnonterminal = 1.0 - next_done
                        next_values = next_value
                        last_gae_lam = 0
                    else:
                        # nextnonterminal = 1.0 - self.dones[t + 1]
                        next_values = self.values[t + 1]
                        last_gae_lam = self.returns[t + 1]

                    delta = (
                        self.rewards[t]
                        + self.args["gamma"]
                        * next_values
                        * -self.values[t]  # nextnonterminal
                    )
                    self.advantages[t] = (
                        delta
                        + self.args["gamma"] * self.args["gae_lambda"]
                        # * nextnonterminal
                        * last_gae_lam
                    )
                self.returns = self.advantages + self.values

            else:
                self.returns[-1] = self.rewards[-1]
                for i in reversed(range(self.args["steps"] - 1)):
                    self.returns[i] = (
                        self.rewards[i] + self.returns[i + 1] * self.args["gamma"]
                    )

                self.advantages = self.returns - self.values

    def PPO_update(self):
        for epoch in range(self.args["epochs"]):
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

                surrogate_loss = -clip_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                surrogate_loss.backward()
                # nn.utils.clip_grad_norm_(self.agent.parameters(), self.args["max_grad_norm"])
                self.optimizer.step()

    def minibatch_generator(self):
        """This generates a shuffled list of minibatches
        When next_episode is called, this will have new values because the self values will be changed
        """

        # NOTE this shuffles but in a Polish way
        # contrary to popular belief, this is 2% faster than shuffle
        batch_idxs = np.random.choice(
            self.args["batch_size"], self.args["batch_size"], replace=False
        )

        # batch_idxs = np.random.shuffle(np.arange(self.args["batch_size"])

        for start in range(0, self.args["batch_size"], self.args["minibatch_size"]):
            end = start + self.args["minibatch_size"]
            minibatch_idxs = batch_idxs[start:end]

            yield (
                self.observations[minibatch_idxs],
                self.logprobs[minibatch_idxs],
                self.actions[minibatch_idxs],
                self.advantages[minibatch_idxs],
                self.values[minibatch_idxs],
                self.returns[minibatch_idxs],
            )

    def reset_memory(self):
        self.observations = torch.zeros(self.args["steps"], self.obs_shape).to(
            self.device
        )
        self.actions = torch.zeros(self.args["steps"])
        # The ActorCritic Network outputs log probabilities
        self.logprobs = torch.zeros_like(self.actions)
        self.rewards = torch.zeros_like(self.logprobs)
        self.values = torch.zeros_like(self.logprobs)
        self.advantages = torch.zeros_like(self.logprobs)
        self.returns = torch.zeros_like(self.logprobs)

    def show_loaded_model(self):
        load_model(self.agent)

        env = gym.make(
            self.args["env_name"], render_mode="human", max_steps=self.args["steps"]
        )

        observation = torch.Tensor(env.reset()[0]["image"].flatten()).to(self.device)

        for step in range(1000):
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(observation)

            observation, reward, done, truncated, info = env.step(
                np.array([action.cpu().numpy()])
            )

            if done:
                observation, _ = env.reset()

            observation = torch.Tensor(observation["image"].flatten()).to(self.device)


if __name__ == "__main__":
    ppo = PPO("hyperparams.json", False)
    # save_params(self.self.args)
    ppo.train()
