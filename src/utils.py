import constants
import numpy as np
from llama import Llama
import os, time, random
from sentence_transformers import SentenceTransformer


base_path = os.path.dirname(__file__)

class llama2_7b_reward_shaper:
    """Reward shaper

    Takes in an observation matrix, and generates a list of suggestions. These can be compared with actions and reward will be generated
    """

    def __init__(self, goal, temperature=0.6, top_p=0.9, rl_temp=0, dialogue_memory=24):
        self.generator = Llama.build(
            ckpt_dir=os.path.join(base_path, "../llama-2-7b-chat"),
            tokenizer_path=os.path.join(
                base_path, "../llama-2-7b-chat/tokenizer.model"
            ),
            max_seq_len=2048,
            max_batch_size=6,
        )

        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.temperature = temperature
        self.dialogue_memory = dialogue_memory
        self.top_p = top_p
        self.rl_temp = rl_temp
        self.suggestions = None


        self.dialog = [
            {
                "role": "system",
                "content": f"You are a helpful assistant giving advice to someone playing a videogame. You will recieve the current goal of the game and a list of observations about the environment, and you should give a list of suggested actions that the player should take to reach their goal. The suggested actions should involve the objects mentioned and the following legal actions: {PICK UP, DROP, TOGGLE}. You should not make assumptions about the environment, only use the information given to you. Separate each suggestion with a new line.",
            },
            {
                "role": "user",
                "content": "My goal is: pick up the purple box. I see a red key and a red door",
            }
        ]

    def suggest(self, observation):
        """Creates a list of suggested actions based on the current observation

        Args:
            observation: the full observation returned by gymnasium environment

        Returns:
            A list of strings of suggested actions by the LLM. Also sets self.suggestions

        """

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

    def compare(self, action, cell):
        """Compares the semantic similarity between an action and the current list of suggested actions

        Args:
            action: action as Discrete() ie. an integer
            cell: ndarray of shape (3,)

        Returns:
            a reward if action and one of the suggested actions is semantically similar
        """

        action = constants.IDX_TO_ACTION[action]
        item = constants.IDX_TO_OBJECT[cell[0]]
        color = constants.IDX_TO_COLOR[cell[1]]
        state = constants.IDX_TO_COLOR[cell[2]] if item == "door" else None





class llama2_7b_policy:
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
                "content": "You are a player playing a videogame. It is a top down turn based game, where each turn you can perform one action. Every action is either a movement, or an interaction. List of possible movement actions: {'RIGHT', 'LEFT', 'FORWARD'}. List of possible interaction actions: {'PICKUP', 'DROP', 'TOGGLE'}. List of interactable objects in the game: {'KEY', 'DOOR', 'CHEST'}. You are to decide which action is performed on the current turn. When answering, you shall strictly only reply with a single one of the capitalized actions from the either the movement action list, or the interaction action list. You must always answer with EXACTLY 1 word.",
            },
            {
                "role": "user",
                "content": "You see a red key 1 square FORWARD. What should you do?",
            },
            {
                "role": "assistant",
                "content": "PICKUP",
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

        elif "PICKUP" in answer:
            action_list.append(constants.ACTION_TO_IDX["pick"])

        elif "TOGGLE" in answer:
            action_list.append(constants.ACTION_TO_IDX["toggle"])

        elif "DROP" in answer:
            action_list.append(constants.ACTION_TO_IDX["drop"])

        else:
            action_list.append(constants.ACTION_TO_IDX["toggle"])

        action_list.append(constants.ACTION_TO_IDX[input("move: ")])

        if len(self.dialog) > self.dialogue_memory:
            self.dialog.pop(3)
            self.dialog.pop(3)



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
            string = f"see a {state} {color} {object_str} {longitude} and {latitude}"

        elif longitude and not latitude:
            string = f"see a {state} {color} {object_str} {longitude}"

        elif not longitude and latitude:
            string = f"see a {state} {color} {object_str} {latitude}"

        elif not longitude and not latitude:
            string = f"have a {state} {color} {object_str} in your inventory"

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

    prompt = f"You {', '.join(observation_strings) if observation_strings else 'nothing'}. What should you do?"
    # Please only answer with a single of the following commands: RIGHT, LEFT, FORWARD or PICK UP."

    return prompt
