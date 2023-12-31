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
import matplotlib.pyplot as plt
import argparse

base_path = os.path.dirname(__file__)


class PPO:
    def __init__(
        self,
        param_file,
        result_file=None,
        reward=None,
        policy=None,
        generator=None,
        allow_farming=False,
        policy_sim_thres=None,
        policy_sim_mod=None,
    ):
        self.args = read_params(param_file)

        # these arguments can overwrite the file if set
        self.results_file = (
            self.args["results_file"] if not result_file else result_file
        )
        self.llama_policy = self.args["llama_policy"] if policy is None else policy
        self.llama_reward = self.args["llama_reward"] if reward is None else reward

        self.env = gym.make(
            self.args["env_name"], render_mode="rgb_array", max_steps=self.args["steps"]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.agent = Agent(self.env, self.llama_policy, generator=generator, policy_sim_thres=policy_sim_thres, policy_sim_mod=policy_sim_mod).to(
            self.device
        )
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args["lr"], eps=1e-8
        )
        self.obs_shape = int(np.array(self.env.observation_space["image"].shape).prod())

        self.reward_shaper = (
            llama2_reward_shaper(
                self.env.reset()[0]["mission"],
                similarity_modifier=0.005,
                cos_sim_threshold=0.84,
                generator=generator,
                allow_farming=allow_farming,
            )
            if self.llama_reward
            else None
        )

        # Initialize all observation-, return-, valuevectors and so on
        self.reset_memory()

    def train(self):
        """Generates samples from a single episode, backpropagates the samples and
        repeats for rollout number of times
        """

        rewards = list()

        self.args["batch_size"] = self.args["steps"]
        self.args["minibatch_size"] = int(
            self.args["batch_size"] // self.args["num_minibatches"]
        )
        optimizer = optim.Adam(self.agent.parameters(), lr=self.args["lr"], eps=1e-5)

        # Initializing the next step

        # Episode: Moving n steps and estimating the value funvtion for each step

        i = 0
        for rollout in range(self.args["rollouts"]):
            print(f"Rollout num: {rollout}")
            (
                env_reward,
                advisor_reward_cum,
            ) = self.next_episode(
                rollout
            )  # Perform steps and store relevant values
            rewards.append(env_reward)
            # if not env_reward:
            #     print("No reward received, skipping update of networks")
            #     continue
            print(f"Env reward: {env_reward}")
            print(f"LLM reward: {advisor_reward_cum}")
            if self.reward_shaper:
                print(f"Cache misses: {self.reward_shaper.cache_misses}")
                self.reward_shaper.cache_misses = 0
            if self.agent.consigliere:
                print(f"Cache misses: {self.agent.consigliere.cache_misses}")
                self.agent.consigliere.cache_misses = 0
            self.PPO_update()  # Use values stored to backpropagate using PPO
            self.reset_memory()  # Restore

            if i % 10 == 0:
                save_model(self.agent)
                if self.results_file:
                    file_name = os.path.join(base_path, self.results_file)
                    if os.path.exists(file_name):
                        os.remove(file_name)
                    np.save(file_name, np.array(rewards))
                if self.reward_shaper:
                    self.reward_shaper.save_cache()
                if self.agent.consigliere:
                    self.agent.consigliere.save_cache()

            i += 1

    def next_episode(self, rollout):
        """Steps through a single episode and calculates what is needed to
        perform PPO update
        """

        # print("Entered next_episode()")
        observation_dict, _ = self.env.reset()
        observation = torch.Tensor(observation_dict["image"].flatten()).to(self.device)
        env_reward = 0
        advisor_reward_cum = 0
        if self.reward_shaper:
            self.reward_shaper.goal = observation_dict["mission"]

        if self.agent.consigliere:
            self.agent.consigliere.goal = observation_dict["mission"]

        for step in range(self.args["steps"]):
            self.observations[step] = observation

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    observation, rollout=rollout
                )

            self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # LLM reward shaping
            if self.reward_shaper:
                # this sets self.suggestions
                self.reward_shaper.suggest(observation_dict["image"])
                # print(action)
                advisor_reward = self.reward_shaper.compare(
                    int(action.cpu()), observation_dict["image"]
                )
            else:
                advisor_reward = 0

            observation_dict, reward, done, truncated, info = self.env.step(
                action.cpu().numpy()
            )
            env_reward += reward
            advisor_reward_cum += advisor_reward

            self.rewards[step] = reward + advisor_reward

            observation = torch.Tensor(observation_dict["image"].flatten()).to(
                self.device
            )

            if done:
                break

        if self.reward_shaper:
            self.reward_shaper.caption_set = set()
            # self.reward_shaper.reset_cache()

        # Now that the agent has played out an episode, it's time
        # to backtrack all steps, and compute the discounted rewards
        with torch.no_grad():
            self.returns[-1] = self.rewards[-1]
            for i in reversed(range(self.args["steps"] - 1)):
                self.returns[i] = (
                    self.rewards[i] + self.returns[i + 1] * self.args["gamma"]
                )
            self.advantages = self.returns - self.values

            # self.advantages = torch.where(
            #     self.returns != 0,
            #     self.returns - self.values,
            #     torch.zeros_like(self.returns),
            # )
            # print(f"{self.advantages=}")
        return env_reward, advisor_reward_cum

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

                surrogate_loss = -clip_loss + value_loss - entropy_loss

                self.optimizer.zero_grad()
                surrogate_loss.backward()
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
        self.actions = torch.zeros(self.args["steps"]).to(self.device)
        # The ActorCritic Network outputs log probabilities
        self.logprobs = torch.zeros_like(self.actions).to(self.device)
        self.rewards = torch.zeros_like(self.logprobs).to(self.device)
        self.values = torch.zeros_like(self.logprobs).to(self.device)
        self.advantages = torch.zeros_like(self.logprobs).to(self.device)
        self.returns = torch.zeros_like(self.logprobs).to(self.device)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--result_file",
        type=str,
        help="Output file for env rewards",
        default=None,
    )
    parser.add_argument(
        "-r", "--reward", type=bool, help="Use LLM reward shaping", default=None
    )
    parser.add_argument(
        "-p", "--policy", type=bool, help="Use LLM policy", default=None
    )
    parser.add_argument("-a", "--allow_farming", action="store_true")
    parser.add_argument("-d", "--doobie", action="store_true")

    args = parser.parse_args()
    print(args)

    if args.doobie:
        generator = 1
    else:
        generator = Llama.build(
            ckpt_dir=os.path.join(base_path, "../llama-2-7b-chat"),
            tokenizer_path=os.path.join(
                base_path, "../llama-2-7b-chat/tokenizer.model"
            ),
            max_seq_len=2048,
            max_batch_size=6,
        )

    ppo = PPO(
        "hyperparams.json",
        result_file=args.result_file,
        reward=args.reward,
        policy=args.policy,
        generator=generator,
        allow_farming=args.allow_farming,
    )
    ppo.train()
