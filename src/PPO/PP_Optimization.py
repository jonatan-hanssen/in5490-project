import argparse
import os
from distutils.util import strtobool
import time
import numpy as np
import random
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from ActorCritic import *
from utils import *
import json


def minibatch_generator():
    batch_idx = np.random.choice(args["batch_size"], args["batch_size"], replace=False)
    for start in range(0, args["batch_size"], args["minibatch_size"]):
        end = start + args["minibatch_size"]


def PPO(param_path, device=torch.device("cpu")):
    args = read_params(param_path)
    args["batch_size"] = int(args["num_envs"] * args["ep_steps"])
    args["minibatch_size"] = int(args["batch_size"] // args["num_minibatches"])
    seeding(args["seed"])
    seeds = [random.randint(0, 20000) for _ in range(args["num_envs"])]
    envs = gym.vector.SyncVectorEnv(
        [make_env(args["env_name"], seeds[i]) for i in range(args["num_envs"])]
    )
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args["lr"], eps=1e-5)
    # _, _ = envs.reset(seed=args["seed"])
    args["rollouts"] = int(args["tot_steps"] // args["batch_size"])

    single_obs_shape = int(
        np.array(envs.single_observation_space["image"].shape).prod()
    )
    # print(envs.single_observation_space)

    # The following tensors need not to be initialized to again
    observations = torch.zeros(
        (args["ep_steps"], args["num_envs"], single_obs_shape),
    ).to(device)
    actions = torch.zeros(
        (args["ep_steps"], args["num_envs"]) + envs.single_action_space.shape
    ).to(device)
    # The ActorCritic Network outputs log probabilities
    logprobs = torch.zeros((args["ep_steps"], args["num_envs"])).to(device)
    rewards = torch.zeros((args["ep_steps"], args["num_envs"])).to(device)
    dones = torch.zeros((args["ep_steps"], args["num_envs"])).to(device)
    values = torch.zeros((args["ep_steps"], args["num_envs"])).to(device)

    # Initializing the next step
    next_observation = torch.Tensor(envs.reset()[0]["image"].flatten()).to(device)
    next_done = torch.zeros(args["num_envs"]).to(device)
    # next_done = torch.zeros(.args.num_envs).to(device)

    # Episode: Moving n steps and estimating the value funvtion for each step
    for rollout in range(args["rollouts"] + 1):
        print(f"Rollout num: {rollout}")
        (
            obs,
            acts,
            logs,
            rews,
            dones,
            vals,
            adv,
            rets,
            next_observation,
            next_done,
        ) = env_episode(
            agent,
            args,
            device,
            envs,
            next_observation,
            next_done,
            rollout,
            observations,
            actions,
            logprobs,
            rewards,
            dones,
            values,
        )
        print(f"Sum of rewards per episode: {torch.sum(rewards)}")
        PPO_update(
            optimizer,
            agent,
            device,
            envs,
            args,
            obs,
            acts,
            logs,
            rews,
            dones,
            vals,
            adv,
            rets,
        )


def env_episode(
    agent,
    args,
    device,
    envs,
    next_observation,
    next_done,
    rollout_num,
    observations,
    actions,
    logprobs,
    rewards,
    dones,
    values,
    gae=True,
):
    for step in range(args["ep_steps"]):
        # global_step += args["num_envs"]
        # if rollout_num == 0:
        # print(envs.reset()[0]["image"].flatten())
        # next_observation = torch.Tensor(envs.reset()[0]["image"].flatten()).to(
        #     device
        # )
        # next_done = torch.zeros(args["num_envs"]).to(device)
        # print(next_observation)
        observations[step] = next_observation
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_observation)
            # print(np.array([action.cpu().numpy()]))
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        next_observation, reward, done, truncated, info = envs.step(
            np.array([action.cpu().numpy()])
        )
        # Converting to tensor and transfering to gpu for faster computation
        rewards[step] = torch.Tensor(reward).to(device).view(-1)

        # print(next_observation["image"])

        next_observation, next_done = (
            torch.Tensor(next_observation["image"].flatten()).to(device),
            torch.Tensor(next_done).to(device),
        )

    # Now that the agent(-s) has(-ve) played out an episode, it's time
    # to backtrack all steps, and compute the discounted rewards
    with torch.no_grad():
        next_value = agent.get_value(next_observation).reshape(1, -1)
        # General advanatage estimation proposed in the original paper on PPO
        if gae:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args["ep_steps"])):
                if t == args["ep_steps"] - 1:
                    # "nextnonterminal" is a environment specific variable indicating if the
                    # agent has finished the game before reaching time step limit
                    nextnonterminal = 1.0 - next_done
                    next_values = next_value
                    last_gae_lam = 0
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                    last_gae_lam = returns[t + 1]

                delta = (
                    rewards[t]
                    + args["gamma"] * next_values * nextnonterminal
                    - values[t]
                )
                returns[t] = (
                    delta
                    + args["gamma"]
                    * args["gae_lambda"]
                    * nextnonterminal
                    * last_gae_lam
                )
            advantages = returns - values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]

                returns[t] = (
                    rewards[t] * args["gamma"] * nextnonterminal * returns[t + 1]
                )
            advantages = returns - values

    return (
        observations,
        actions,
        logprobs,
        rewards,
        dones,
        values,
        advantages,
        returns,
        next_observation,
        next_done,
    )


def PPO_update(
    optimizer,
    agent,
    device,
    envs,
    args,
    observations,
    actions,
    logprobs,
    rewards,
    dones,
    values,
    advantages,
    returns,
):
    single_obs_shape = int(
        np.array(envs.single_observation_space["image"].shape).prod()
    )
    # Optimization of the surrogate loss

    # Flattening all containers in order to compute components of surrogate losses
    # using minibatches
    batch_observations = observations.view((-1, single_obs_shape))
    batch_logprobs = logprobs.view(-1)
    batch_actions = actions.view((-1,) + envs.single_action_space.shape)
    batch_advantages = advantages.view(-1)
    batch_values = values.view(-1)
    batch_returns = returns.view(-1)

    # batch_idx = np.arange(args["batch_size"])
    for epoch in range(args["epochs"]):
        # np.random.shuffle(batch_idx)
        # NOTE this shuffles but in a greeeg way
        batch_idx = np.random.choice(
            args["batch_size"], args["batch_size"], replace=False
        )
        for start in range(0, args["batch_size"], args["minibatch_size"]):
            end = min(start + args["minibatch_size"], args["batch_size"])
            end = start + args["minibatch_size"]
            minibatch_idx = batch_idx[start:end]
            # print(minibatch_idx)

            _, new_logprob, entropy, new_value = agent.get_action_and_value(
                batch_observations[minibatch_idx], batch_actions[minibatch_idx]
            )

            # The probability ratio of the new policy vs the old policy
            # this is equivalent to prob / prob_old
            policy_ratio = (new_logprob - batch_logprobs.long()[minibatch_idx]).exp()

            minibatch_advantages = batch_advantages[minibatch_idx]

            p_grad_clip1 = policy_ratio * minibatch_advantages
            p_grad_clip2 = minibatch_advantages * torch.clamp(
                policy_ratio,
                1.0 - args["clip_epsilon"],
                1.0 + args["clip_epsilon"],
            )
            # Whenever a paper writes an equation as the expected value, take the mean() of that function
            # to imitate the expected value
            clip_loss = torch.min(p_grad_clip1, p_grad_clip2).mean()

            # Computing Value Loss
            value_loss = (
                (new_value.view(-1) - batch_returns[minibatch_idx]) ** 2
            ).mean() * args["value_loss_coef"]

            # Computing expected value of Entropy Loss
            entropy_loss = entropy.mean() * args["entropy_coef"]

            surrogate_loss = clip_loss - value_loss + entropy_loss
            optimizer.zero_grad()
            surrogate_loss.backward()
            nn.utils.clip_grad_norm(agent.parameters(), args["max_grad_norm"])
            optimizer.step()


def seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save_params(params)
    PPO("hyperparams.json", device)
