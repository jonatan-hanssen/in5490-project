import constants
import numpy as np
from gradio_client import Client
import random
import time


class llama2_70b_policy:
    def __init__(self, temperature=0.1):
        self.client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")
        self.temperature = temperature

    def __call__(self, prompt, action_list, env):
        if "nothing" in prompt:
            action_list.append(env.action_space.sample())
            return

        heat = random.random()
        print(f"{heat=}")

        if heat < self.temperature:
            print("here")
            action_list.append(env.action_space.sample())
            return

        answer = self.client.predict(prompt, api_name="/chat")

        print(answer)

        if "error" in answer or "ERROR" in answer:
            time.sleep(3)
            self.client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")
            action_list.append(env.action_space.sample())
            return

        if "FORWARD" in answer:
            action_list.append(2)

        elif "LEFT" in answer:
            action_list.append(0)
            action_list.append(2)

        elif "RIGHT" in answer:
            action_list.append(1)
            action_list.append(2)

        elif "PICK" in answer:
            action_list.append(3)


def obs_to_string(obs_matrix):
    key_inds = np.where(obs_matrix[:, :, 0] == constants.OBJECT_TO_IDX["key"])
    key_inds = np.stack(key_inds, axis=-1)

    door_inds = np.where(obs_matrix[:, :, 0] == constants.OBJECT_TO_IDX["door"])
    door_inds = np.stack(door_inds, axis=-1)

    wall_inds = np.where(obs_matrix[:, :, 0] == constants.OBJECT_TO_IDX["wall"])
    wall_inds = np.stack(wall_inds, axis=-1)

    def ind_to_string(ind, object_str):
        rel_ind = ind - [3, 6]

        color = constants.IDX_TO_COLOR[obs_matrix[ind[0], ind[1], 1]]

        if object_str == "door":
            state = constants.IDX_TO_STATE[obs_matrix[ind[0], ind[1], 2]]
        else:
            state = ""

        if rel_ind[0] == 0:
            longitude = ""

        elif rel_ind[0] > 0:
            longitude = f"{rel_ind[0]} square{'s' if rel_ind[0] != 1 else ''} RIGHT"

        else:
            longitude = (
                f"{abs(rel_ind[0])} square{'s' if abs(rel_ind[0]) != 1 else ''} LEFT"
            )

        if rel_ind[1] == 0:
            latitude = ""

        else:
            latitude = (
                f"{abs(rel_ind[1])} square{'s' if abs(rel_ind[1]) != 1 else ''} FORWARD"
            )

        if longitude and latitude:
            string = f"a {state} {color} {object_str} {longitude} and {latitude}"

        elif longitude and not latitude:
            string = f"a {state} {color} {object_str} {longitude}"

        elif not longitude and latitude:
            string = f"a {state} {color} {object_str} {latitude}"

        elif not longitude and not latitude:
            string = f"a {state} {color} {object_str} in your inventory"

        return string

    observation_strings = list()

    for ind in key_inds:
        observation_strings.append(ind_to_string(ind, "key"))

    for ind in door_inds:
        observation_strings.append(ind_to_string(ind, "door"))

    for ind in wall_inds:
        if ind[0] - 3 == 0 and ind[1] - 6 == -1:
            observation_strings.append(ind_to_string(ind, "wall"))

    prompt = f"You are a player playing a videogame. It is a top down turn based game, where each turn you can either move RIGHT, LEFT, FORWARD or PICK UP objects 1 square in front of you. You see {', '.join(observation_strings) if observation_strings else 'nothing'}. What should you do? Please only answer with a single of the following commands: RIGHT, LEFT, FORWARD or PICK UP."

    return prompt
