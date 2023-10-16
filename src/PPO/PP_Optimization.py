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
import json


def read_params():
    file = open("hyperparams.json")
    params = json.load(file)
    # print(json.dumps(params, indent=4, separators=(":", ",")))
    return params


def save_params(params):
    file = open("hyperparams.json", "w")
    json.dump(params, file, indent=4, separators=(",", ":"))


def sample_minibatches():
    pass


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

    def __init__(, envs):
        super(Agent, ).__init__()
        .critic = nn.Sequential(
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

        .actor = nn.Sequential(
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

    def get_value(, X):
        return .critic(X)

    def get_action_and_value(, X, action=None):
        logits = .actor(X)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), .critic(X)


def PPO(param_path):
    args = read_params()
    seeding()
    seeds = [random.randint(0, 20000) for _ in range(args["env_n"])]
    envs = gym.vector.SyncVector(
        [make_env(env_name, seeds[i]) for i in range(args["num_envs"])]
    )
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    _, _ = envs.reset(seed=args["seed"])
    args["rollouts"] = args["tot_steps"] // args["batch_size"]


def save_model(model):
    file_path = os.path.dirname(__file__) + "/model.zip"
    torch.save(model.state_dict(), file_path)


def load_model(model):
    file_path = os.path.dirname(__file__) + "/model.zip"
    model.load_state_dict(torch.load(file_path))


def PPO_train(envs):
    # Containers for values needed in calculation of surrogate loss
    single_obs_shape = (
        int(np.array(envs.single_observation_space["image"].shape).prod()),
    )

    observations = torch.zeros(
        (args["ep_steps"], args["num_envs"], single_obs_shape),
    ).to(device)
    actions = torch.zeros(
        (args["ep_steps"], args["num_envs"]) + envs.single_action_space
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
        env_episode(rollout)
        comp_advantage(gae=args["gae"])
        PPO_update()


def env_episode(rollout_num, agent):
    for step in range(args["ep_steps"]):
        global_step += args["num_envs"]
        if rollout_num == 0:
            observations[step] = torch.zeros(envs.reset()[0]).to(device)
            dones[step] = torch.zeros(args["num_envs"]).to(device)

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_observation)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        next_observation, reward, done, truncated, info = envs.step(
            action.cpu().numpy()
        )
        # Converting to tensor and transfering to gpu for faster computation
        rewards[step] = torch.Tensor(reward).to(device).view(-1)

        next_observation, next_done = (
            torch.Tensor(next_observation).to(device),
            torch.Tensor(next_done).to(device),
        )

def comp_advantage(gae=False):
    # Now that the agent(-s) has(-ve) played out an episode, it's time
    # to backtrack all steps, and compute the discounted rewards
    with torch.no_grad():
        next_value = agent.get_value(next_observation).reshape(1, -1)
        # General advanatage estimation proposed in the original paper on PPO
        if gae:
            returns = torch.zeros_like(rewards).to(device)
            # lastgaelam = 0
            for t in reversed(range(args["num_steps"])):
                if t == args["num_steps"] - 1:
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
            advantages = ret  values
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
                    rewards[t]
                    * args["gamma"]
                    * nextnonterminal
                    * returns[t + 1]
                )
            advantages = returns - values


def backward():
    # Optimization of the surrogate loss

    # Flattening all containers in order to compute components of surrogate losses
    # using minibatches
    batch_observations = .observations.view(
        (-1,) + .single_obs_shape
    )  # if "view(-1) does not work, use reshape()
    batch_logprobs = .logprobs.view(-1)
    batch_actions = .actions.view((-1,) + envs.single_action_space.shape)
    batch_advantages = .advantages.view(-1)
    batch_values = .values.view(-1)
    batch_returns = .returns.view(-1)

    # batch_idx = np.arange(.args.input_size)
    for epoch in range(.args.epochs):
        # Shuffle the batch indexes to introduce additional noise and variance
        # np.random.shuffle(batch_idx)
        # TODO: Implement a minibatch generator here -> Gregz
        batch_idx = np.random.choice(
            args["batch_size"], args["batch_size"], replace=False
        )
        for start in range(0, args["batch_size"], args.["minibatch_size"]):
            end = start + args["batch_size"]
            minibatch_idx = batch_idx[start:end]

            newAction, newLogprob, entropy, newValue = agent.get_action_and_value(
                batch_observations[minibatch_idx],
                batch_actions.long()[minibatch_idx],
            )

            # The probability ratio of the new policy vs the old policy
            policy_ratio = (newLogprob - batch_logprobs[minibatch_idx]).exp()

            minibatch_advantages = batch_advantages[minibatch_idx]
            # TODO: Add normalization of minibatch_advantage

            pGrad_clip1 = policy_ratio * minibatch_advantages
            pGrad_clip2 = minibatch_advantages * torch.clamp(
                policy_ratio,
                1.0 - args["clip_epislon"],
                1.0 + args["clip_epislon"],
            )
            # Whenever a paper writes an equation as the expected value, take the mean() of that function
            # to imitate the expected value
            clip_Loss = torch.min(pGrad_clip1, pGrad_clip2).mean()

            # Computing Value Loss
            value_loss = (
                (1 / 2)
                * ((newValue.view(-1) - batch_values[minibatch_idx]) ** 2).mean()
                * args["coef1"]
            )

            # Computing Entropy Loss
            entropy_loss = entropy.mean() * args["coef2"]

            surrogate_loss = clip_Loss - value_loss + entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(
                agent.parameters(), args["max_grad_norm"]
            )
            .optimizer.step()

def seeding():
    random.seed(.args.seed)
    np.random.seed(.args.seed)
    torch.manual_seed(.args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print("The Main function")

    read_params()
