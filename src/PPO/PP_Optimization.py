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
import json


def read_params(params):
    file = open(params)
    params = json.load(file)
    # print(json.dumps(params, indent=4, separators=(":", ",")))
    return params


def save_params(params):
    file = open("hyperparams.json", "w")
    json.dump(params, file, indent=4, separators=(",", ":"))


def minibatch_generator():
    batch_idx = np.random.choice(args["batch_size"], args["batch_size"], replace=False)
    for start in range(0, args["batch_size"], args["minibatch_size"]):
        end = start + args["minibatch_size"]


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
    """
    Descr:
        Returns a "vectorized" environenment, meaning that it is wrapped in a gymnasium vector, allowing for
        some aditional functionality, such as parallel training of many environments.
    """

    def env_gen():
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return env_gen


# TODO: Transfer model to device a.k.a. to GPU
class Agent(nn.Module):
    """
    Descr:
        The agents task is to decide the next action to perform, and evaluate the
        possible future rewards, a.k.a. the value function, based on the current state
        the agent is in.
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
        # .LM_actor = llama2_7b_policy()

    def get_value(self, X):
        return self.critic(X)

    def get_action_and_value(self, X, action=None):
        logits = self.actor(X)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(X)


def PPO(param_path, device=torch.device("cpu")):
    args = read_params(param_path)
    args["batch_size"] = args["num_envs"] * args["tot_steps"]
    seeding(args["seed"])
    seeds = [random.randint(0, 20000) for _ in range(args["num_envs"])]
    envs = gym.vector.SyncVectorEnv(
        [make_env(args["env_name"], seeds[i]) for i in range(args["num_envs"])]
    )
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args["lr"], eps=1e-5)
    _, _ = envs.reset(seed=args["seed"])
    args["rollouts"] = args["tot_steps"] // args["batch_size"]

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
    # next_observation = torch.zeros(.envs.reset()[0]).to(device)
    # next_done = torch.zeros(.args.num_envs).to(device)

    # Episode: Moving n steps and estimating the value funvtion for each step
    for rollout in range(args["rollouts"] + 1):
        obs, acts, logs, rews, dones, vals, adv, rets = env_episode(
            agent,
            args,
            device,
            envs,
            rollout,
            observations,
            actions,
            logprobs,
            rewards,
            dones,
            values,
        )
        PPO_update(optimizer, agent, device, obs, acts, logs, rews, dones, vals, adv, rets)


def checkpoint():
    # TODO: Implement a function for saving model when the perforamce reaches a certain level.
    # Intention is to use the "save_model(model)" method
    raise NotImplementedError()


def save_model(model):
    file_path = os.path.dirname(__file__) + "/model.zip"
    torch.save(model.state_dict(), file_path)


def load_model(model):
    file_path = os.path.dirname(__file__) + "/model.zip"
    model.load_state_dict(torch.load(file_path))


def env_episode(
    agent,
    args,
    device,
    envs,
    rollout_num,
    observations,
    actions,
    logprobs,
    rewards,
    dones,
    values,
    gae=True
):
    for step in range(args["ep_steps"]):
        # global_step += args["num_envs"]
        if rollout_num == 0:
            # print(envs.reset()[0]["image"].flatten())
            next_observation = torch.Tensor(envs.reset()[0]["image"].flatten()).to(device)
            next_done = torch.zeros(args["num_envs"]).to(device) 

        observations[step] = next_observation
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_observation)
            # print(np.array([action.cpu().numpy()]))
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob
        
        next_observation, reward, done, truncated, info = envs.step(np.array([action.cpu().numpy()]))
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
            # lastgaelam = 0
            for t in reversed(range(args["ep_steps"])):
                if t == args["ep_steps"] - 1:
                    # "nextnonterminal" is a environment specific variable indicating if the
                    # agent has finished the game before reaching time step limit
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]

                delta = (
                    rewards[t]
                    + args["gamma"] * values[t + 1] * nextnonterminal
                    - values[t]
                )
                returns[t] = (
                    delta
                    + args["gamma"]
                    * args["gae_lambda"]
                    * nextnonterminal
                    * returns[t + 1]
                )
            advantages = returns - values
            # Standard advantage estimation of General Advantage Estimation
            # proposed by the authors of PPO is not used
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

    return observations, actions, logprobs, rewards, dones, values, advantage, returns


def PPO_update(
    optimizer,
    agent,
    device,
    observations,
    actions,
    logprobs,
    rewards,
    dones,
    values,
    advantage,
    returns,
):
    # Optimization of the surrogate loss

    # Flattening all containers in order to compute components of surrogate losses
    # using minibatches
    batch_observations = observations.view(
        (-1,) + single_obs_shape
    )  # if "view(-1) does not work, use reshape()
    batch_logprobs = logprobs.view(-1)
    batch_actions = actions.view((-1,) + envs.single_action_space.shape)
    batch_advantages = advantages.view(-1)
    batch_values = values.view(-1)
    batch_returns = returns.view(-1)

    # batch_idx = np.arange(.args.input_size)
    for epoch in range(args["epochs"]):
        # NOTE this shuffles but in a greeeg way
        batch_idx = np.random.choice(
            args["batch_size"], args["batch_size"], replace=False
        )
        for start in range(0, args["batch_size"], args["minibatch_size"]):
            end = min(start + args["minibatch_size"], args["batch_size"])
            minibatch_idx = batch_idx[start:end]

            _, new_logprob, entropy, new_value = agent.get_action_and_value(
                batch_observations[minibatch_idx],
                batch_actions.long()[minibatch_idx],
            )

            # The probability ratio of the new policy vs the old policy
            # this is equivalent to prob / prob_old
            policy_ratio = (new_logprob - batch_logprobs[minibatch_idx]).exp()

            minibatch_advantages = batch_advantages[minibatch_idx]
            # TODO: Add normalization of minibatch_advantage

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
            ).mean() * args["coef1"]

            # Computing Entropy Loss
            entropy_loss = entropy.mean() * args["coef2"]

            surrogate_loss = clip_loss - value_loss + entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(agent.parameters(), args["max_grad_norm"])
            optimizer.step()


def seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        "num_envs": 1,
        "tot_steps": 16384,
        "seed": 42069,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "num_minibatches": 4,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "env_name": "MiniGrid-UnlockPickup-v0",
        "ep_steps": 1024,
        "lr": 0.03,
        "epochs": 20,
        "rollouts": 20,
        "rollot_steps": 256,
    }
    save_params(params)
    PPO("hyperparams.json", device)
