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

    def __init__(self, goal, temperature=0.6, top_p=0.9, rl_temp=0):
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
        self.top_p = top_p
        self.rl_temp = rl_temp
        self.suggestions = None
        self.goal = goal


        self.dialog = [
            {
                "role": "system",
                "content": f"You are a helpful assistant giving advice to someone playing a videogame. You will recieve the current goal of the game and a list of observations about the environment, and you should give a list of suggested actions that the player should take to reach their goal. The suggested actions should involve the objects mentioned. The player can pick up items, drop items and use items. You should not make assumptions about the environment, only use the information given to you. If none of the observations are relevant to solving the task, respond that the player should explore more. Separate each suggestion with a new line.",
            },
            {
                "role": "user",
                "content": "My goal is: pick up the purple box. I see a red key and a locked red door. What actions do you suggest?",
            },
            {
                "role": "assistant",
                "content": "Pick up the red key. \nUse the red key on the locked red door.",
            },
            {
                "role": "user",
                "content": "My goal is: open the purple door. I see a red key, an open red door, a purple key, a locked purple door and a green key. What actions do you suggest?",
            },
            {
                "role": "assistant",
                "content": "Pick up the purple key. \nOpen the locked purple door with the purple key.",
            },
            {
                "role": "user",
                "content": "My goal is: open the purple door. I see a green key and a locked green door. What actions do you suggest?",
            },
            {
                "role": "assistant",
                "content": "You should explore more.",
            },
            {
                "role": "user",
                "content": "My goal is: open the purple door. I see a locked purple door and have a purple key in my inventory. What actions do you suggest?",
            },
            {
                "role": "assistant",
                "content": "Use the purple key on the locked purple door.",
            },
        ]

    def observation_caption(self, obs_matrix):

        observation_strings = list()
        for i in range(len(obs_matrix)):
            for j in range(len(obs_matrix[i])):
                cell = obs_matrix[i][j]

                if cell[0] > 3:
                    item = constants.IDX_TO_OBJECT[cell[0]]
                    color = constants.IDX_TO_COLOR[cell[1]]
                    state = f" {constants.IDX_TO_STATE[cell[2]]}" if item == "door" else ""

                    if i == 3 and j == 6:
                        observation_strings.append(f"have a{state} {color} {item} in my inventory")
                    else:
                        observation_strings.append(f"see a{state} {color} {item}")


        observation = f"My goal is: {self.goal}. I {', '.join(observation_strings) if observation_strings else 'see nothing interesting'}. What actions do you suggest?"

        return observation




    def suggest(self, observation):
        """Creates a list of suggested actions based on the current observation

        Args:
            observation: the full observation returned by gymnasium environment

        Returns:
            A list of strings of suggested actions by the LLM. Also sets self.suggestions

        """

        self.dialog.append({"role": "user", "content": observation})

        result = self.generator.chat_completion(
            [self.dialog],  # type: ignore
            max_gen_len=100,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        answer = result[0]["generation"]["content"]

        print(answer)

        self.dialog.pop(-1)

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

        if len(self.dialog) > self.dialogue_memory:
            self.dialog.pop(3)
            self.dialog.pop(3)



def obs_to_string(obs_matrix):

    observation_strings = list()
    for i in range(len(obs_matrix)):
        for j in range(len(obs_matrix[i])):
            cell = obs_matrix[i][j]

            if cell[0] < 4:
                continue

            item = constants.IDX_TO_OBJECT[cell[0]]
            color = constants.IDX_TO_COLOR[cell[1]]
            state = f" {constants.IDX_TO_STATE[cell[2]]}" if item == "door" else ""

            rel_x = i - 3
            rel_y = j - 6

            if rel_x == 0:
                longitude = ""

            elif rel_x > 0:
                longitude = f"{rel_x} square{'s' if rel_x != 1 else ''} RIGHT"

            else:
                longitude = (
                    f"{abs(rel_x)} square{'s' if abs(rel_x) != 1 else ''} LEFT"
                )

            if rel_y == 0:
                latitude = ""

            else:
                latitude = (
                    f"{abs(rel_y)} square{'s' if abs(rel_y) != 1 else ''} FORWARD"
                )

            if longitude and latitude:
                string = f"see a {state} {color} {item} {longitude} and {latitude}"

            elif longitude and not latitude:
                string = f"see a {state} {color} {item} {longitude}"

            elif not longitude and latitude:
                string = f"see a {state} {color} {item} {latitude}"

            elif not longitude and not latitude:
                string = f"have a {state} {color} {item} in your inventory"

            observation_strings.append(string)

    prompt = f"You {', '.join(observation_strings) if observation_strings else 'see nothing interesting'}. What should you do?"
    # Please only answer with a single of the following commands: RIGHT, LEFT, FORWARD or PICK UP."

    return prompt
