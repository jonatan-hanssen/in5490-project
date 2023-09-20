import constants
import numpy as np
from llama import Llama
import os, time, random

base_path = os.path.dirname(__file__)


class llama2_70b_policy:
    def __init__(self, temperature=0.6, top_p=0.9, rl_temp=0, dialogue_memory=24):
        self.generator = Llama.build(
            ckpt_dir=os.path.join(base_path, "../llama-2-7b-chat"),
            tokenizer_path=os.path.join(
                base_path, "../llama-2-7b-chat/tokenizer.model"
            ),
            max_seq_len=2048,
            max_batch_size=6,
        )

        self.temperature = temperature
        self.dialogue_memory = dialogue_memory
        self.top_p = top_p
        self.rl_temp = rl_temp
        self.prev_obs_matrix = np.zeros((7, 7, 3))
        self.dialog = [
            {
                "role": "system",
                "content": "You are a player playing a videogame. It is a top down turn based game, where each turn you can either move RIGHT, LEFT, FORWARD, PICK UP or DROP objects, or TOGGLE objects in front of you. When answering, please only reply with a single of the capitalized commands.",
            }
        ]

    def __call__(self, obs_matrix, action_list, env):
        if np.array_equal(obs_matrix, self.prev_obs_matrix):
            observation = "Nothing changed based on your previous move"
        else:
            observation = obs_to_string(obs_matrix)

        self.prev_obs_matrix = obs_matrix

        # if "nothing" in observation:
        #     action_list.append(env.action_space.sample())
        #     return

        heat = random.random()

        if heat < self.rl_temp:
            action_list.append(env.action_space.sample())
            return

        self.dialog.append({"role": "user", "content": observation})
        print(self.dialog)

        result = self.generator.chat_completion(
            [self.dialog],  # type: ignore
            max_gen_len=100,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        answer = result[0]["generation"]["content"]

        self.dialog.append({"role": "assistant", "content": answer})

        print(answer)

        if "FORWARD" in answer:
            action_list.append(constants.ACTION_TO_IDX["forward"])

        elif "LEFT" in answer:
            action_list.append(constants.ACTION_TO_IDX["left"])
            action_list.append(constants.ACTION_TO_IDX["forward"])

        elif "RIGHT" in answer:
            action_list.append(constants.ACTION_TO_IDX["right"])
            action_list.append(constants.ACTION_TO_IDX["forward"])

        elif "PICK" in answer:
            action_list.append(constants.ACTION_TO_IDX["pick"])

        elif "TOGGLE" in answer:
            action_list.append(constants.ACTION_TO_IDX["toggle"])

        elif "DROP" in answer:
            action_list.append(constants.ACTION_TO_IDX["drop"])

        else:
            action_list.append(constants.ACTION_TO_IDX["toggle"])

        if len(self.dialog) > self.dialogue_memory:
            self.dialog.pop(1)
            self.dialog.pop(1)



def obs_to_string(obs_matrix):
    key_inds = np.where(obs_matrix[:, :, 0] == constants.OBJECT_TO_IDX["key"])
    key_inds = np.stack(key_inds, axis=-1)

    door_inds = np.where(obs_matrix[:, :, 0] == constants.OBJECT_TO_IDX["door"])
    door_inds = np.stack(door_inds, axis=-1)

    wall_inds = np.where(obs_matrix[:, :, 0] == constants.OBJECT_TO_IDX["wall"])
    wall_inds = np.stack(wall_inds, axis=-1)

    box_inds = np.where(obs_matrix[:, :, 0] == constants.OBJECT_TO_IDX["box"])
    box_inds = np.stack(box_inds, axis=-1)

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

    for ind in box_inds:
        observation_strings.append(ind_to_string(ind, "box"))

#    for ind in wall_inds:
#        if ind[0] - 3 == 0 and ind[1] - 6 == -1:
#            observation_strings.append(ind_to_string(ind, "impassable wall"))

    prompt = f"You see {', '.join(observation_strings) if observation_strings else 'nothing'}. What should you do?"
    # Please only answer with a single of the following commands: RIGHT, LEFT, FORWARD or PICK UP."

    return prompt
