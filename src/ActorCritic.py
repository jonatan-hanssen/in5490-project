import os

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from utils import llama2_policy


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


def save_model(model):
    file_path = os.path.dirname(__file__) + "/model.pt"
    torch.save(model.state_dict(), file_path)


def load_model(model):
    file_path = os.path.dirname(__file__) + "/model.pt"
    model.load_state_dict(torch.load(file_path))


def checkpoint():
    # TODO: Implement a function for saving model when the perforamce reaches a certain level.
    # Intention is to use the "save_model(model)" method
    raise NotImplementedError()


# TODO: Transfer model to device a.k.a. to GPU
class Agent(nn.Module):
    """
    Descr:
        The agents task is to decide the next action to perform, and evaluate the
        possible future rewards, a.k.a. the value function, based on the current state
        the agent is in.
    """

    def __init__(self, env, llama=False, consigliere=None):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            init_weightsNbias(nn.Linear(147 * 11, 64, dtype=torch.float64)),
            nn.Tanh(),
            init_weightsNbias(nn.Linear(64, 64, dtype=torch.float64)),
            nn.Tanh(),
            init_weightsNbias(nn.Linear(64, 1, dtype=torch.float64), std=1.0),
        )

        self.actor = nn.Sequential(
            init_weightsNbias(nn.Linear(147 * 11, 64, dtype=torch.float64)),
            nn.Tanh(),
            init_weightsNbias(nn.Linear(64, 64, dtype=torch.float64)),
            nn.Tanh(),
            init_weightsNbias(
                nn.Linear(64, env.action_space.n, dtype=torch.float64), std=0.01
            ),
        )

        goal = env.reset()[0]["mission"]
        if not consigliere:
            self.consigliere = llama2_policy(goal, cos_sim_threshold=0.7, similarity_modifier=1) if llama else None
        else:
            self.consigliere = consigliere


    def get_value(self, observation):
        observation = nn.functional.one_hot(
            observation.to(torch.int64), num_classes=11
        ).flatten()
        return self.critic(observation.to(torch.float64))

    def get_action_and_value(self, observation, action=None, rollout=None):
        observation_onehot = (
            nn.functional.one_hot(observation.to(torch.int64), num_classes=11)
            .to(torch.float64)
            .flatten(start_dim=-2)
        )
        logits = self.actor(observation_onehot)

        if self.consigliere:
            #print(observation.shape)
            # logits /= torch.norm(logits)
            if len(observation.shape) == 2:
                advisor_values_list = list()
                for single_obs in observation:
                    # its flattened so we need to make it normal again
                    unflat_obs = single_obs.reshape((7, 7, 3)).to(torch.int64)
                    # this stores suggested actions
                    # self.consigliere.suggest(unflat_obs)
                    # compares all possible actions to suggestions
                    advisor_values_list.append(self.consigliere.give_values(np.array(unflat_obs.cpu())))
                advisor_values = torch.stack(advisor_values_list)
            else:
                unflat_obs = observation.reshape((7, 7, 3)).to(torch.int64)
                #self.consigliere.suggest(unflat_obs)
                advisor_values = self.consigliere.give_values(np.array(unflat_obs.cpu())).to(torch.float64)

            # print(f"{logits=}")
            # print(f"{advisor_values=}")
            # # anti adrian propaganda
            # logits += advisor_values
            # adrian good vote adrian
            # logits *= advisor_values

        probs = Categorical(logits=logits)

        # print(probs.probs)

        if action is None:
            if self.consigliere:
                if torch.norm(advisor_values) == 0:
                    action = probs.sample()
                else:
                    max_rollout = 500
                    if rollout:
                        anneal = ((max_rollout - rollout) / max_rollout) ** 2
                        if rollout > max_rollout:
                            anneal = 0
                    else:
                        anneal = 1
                    # print(anneal)
                    scale_factor = (torch.norm(logits) / torch.norm(advisor_values))
                    # print(f"{advisor_values=}")
                    # print(f"{scale_factor=}")
                    # print(f"{advisor_values * scale_factor=}")
                    # print(f"{logits=}")
                    advisor_values = advisor_values.to(torch.device("cuda"))
                    newprobs = Categorical(logits=logits + advisor_values * scale_factor * anneal)
                    action = newprobs.sample()
                    # print(f"{probs.probs=}")
                    # print(f"{newprobs.probs=}")
            else:
                action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(observation_onehot),
        )
