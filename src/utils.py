import constants
import torch
import numpy as np
from llama import Llama
import os, time, random, json
from sentence_transformers import SentenceTransformer


base_path = os.path.dirname(__file__)


class llama2_base:
    """Reward shaper

    Takes in an observation matrix, and generates a list of suggestions. These can be compared with actions and reward will be generated
    """

    def __init__(
        self,
        goal,
        cos_sim_threshold=0.6,
        similarity_modifier=0.001,
        temperature=0.6,
        top_p=0.9,
        rl_temp=0,
    ):
        self.generator = Llama.build(
            ckpt_dir=os.path.join(base_path, "../llama-2-7b-chat"),
            tokenizer_path=os.path.join(
                base_path, "../llama-2-7b-chat/tokenizer.model"
            ),
            max_seq_len=2048,
            max_batch_size=6,
        )

        self.similarity_modifier = similarity_modifier
        self.cos_sim_threshold = cos_sim_threshold

        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.temperature = temperature
        self.top_p = top_p
        self.rl_temp = rl_temp
        self.suggestions = None
        self.goal = goal

        self.cache = dict()

        self.caption_set = set()

        self.dialog = [
            {
                "role": "system",
                "content": f"You are a helpful assistant giving advice to someone playing a videogame. You will recieve the current goal of the game and a list of observations about the environment, and you should give a list of suggested actions that the player should take to reach their goal. The suggested actions should involve the objects mentioned. The player can pick up items, drop items and use items. You should not make assumptions about the environment, only use the information given to you. If none of the observations are relevant to solving the task, respond that the player should explore more. Separate each suggestion with a new line. Be concise.",
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

    def suggest(self, observation):
        """Creates a list of suggested actions based on the current observation

        Args:
            observation: image observation returned by gym environment

        Returns:
            A list of strings of suggested actions by the LLM. Also sets self.suggestions

        """
        observation = obs_to_string(observation, False, False)

        observation = f"My goal is: {self.goal}. {observation}"

        self.dialog.append({"role": "user", "content": observation})


        if observation in self.cache.keys():
            answer = self.cache[observation]

        else:
            # print("Cache miss")
            result = self.generator.chat_completion(
                [self.dialog],  # type: ignore
                max_gen_len=100,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            answer = result[0]["generation"]["content"]

            # print(f"{observation=}")
            # print(f"{answer=}")

            self.cache[observation] = answer

        self.dialog.pop(-1)

        self.suggestions = answer

    def reset_cache(self):
        self.cache = dict()

    def compare(self, action, obs_matrix):
        """Compares the semantic similarity between an action and the current list of suggested actions

        Args:
            action: action as Discrete() ie. an integer
            obs_matrix: observation matrix 7x7x3

        Returns:
            a reward if action and one of the suggested actions is semantically similar
        """
        caption = caption_action(action, obs_matrix)

        if not caption or caption in self.caption_set:
            return 0

        self.caption_set.add(caption)

        max_cos_sim = 0
        # print(f"{caption=}")
        for suggestion in self.suggestions.splitlines():
            # print(f"{suggestion=}")
            a, b = self.semantic_model.encode([caption, suggestion])

            cos_sim = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

            if cos_sim > max_cos_sim:
                max_cos_sim = cos_sim

        if max_cos_sim > self.cos_sim_threshold:
            return cos_sim * self.similarity_modifier
        else:
            return 0


def caption_action(action, obs_matrix):
    """takes in an action and the observation and returns a string caption"""

    front_cell = obs_matrix[3, 5]
    inventory = obs_matrix[3, 6]

    action = constants.IDX_TO_ACTION[action]
    front_item = constants.IDX_TO_OBJECT[front_cell[0]]
    front_color = constants.IDX_TO_COLOR[front_cell[1]]
    front_state = (
        f" {constants.IDX_TO_STATE[front_cell[2]]}" if front_item == "door" else ""
    )

    inventory_item = constants.IDX_TO_OBJECT[inventory[0]]
    inventory_color = constants.IDX_TO_COLOR[inventory[1]]
    inventory_state = (
        f" {constants.IDX_TO_COLOR[inventory[2]]}" if inventory_item == "door" else ""
    )

    ###  Captioning action based on cell and inventory ###

    caption = "do nothing"

    if action == "done":
        caption = "mark the game as completed"

    if action == "forward":
        if front_item == "empty":
            caption = "move forward"
        elif front_item == "door" and front_state == " open":
            caption = "move forward"

    if action == "right":
        caption = "turn right"

    if action == "left":
        caption = "turn left"

    if action == "pick":
        if front_cell[0] < 4:  # nothing to pick up
            pass

        elif inventory[0] > 3:  # inventory already full
            pass

        else:
            caption = f"pick up{front_state} {front_color} {front_item}"

    elif action == "drop":
        if inventory[0] < 4:  # nothing to drop
            pass

        elif front_item != "empty":  # cannot drop onto occupied tile
            pass

        else:
            caption = f"drop{inventory_state} {inventory_color} {inventory_item}"

    elif action == "toggle":
        if front_item == "box":
            caption = f"destroy {front_color} box"
        else:
            if front_cell[0] >= 4:
                caption = f"use{front_state} {front_color} {front_item}"

            if inventory[0] > 3:
                caption += f" with{inventory_state} {inventory_color} {inventory_item}"

    return caption


class llama2_reward_shaper(llama2_base):
    def __init__(
        self,
        goal,
        cos_sim_threshold=0.6,
        similarity_modifier=0.001,
        temperature=0.6,
        top_p=0.9,
        rl_temp=0,
    ):
        super().__init__(
            goal, cos_sim_threshold, similarity_modifier, temperature, top_p, rl_temp
        )


class llama2_policy(llama2_base):
    def __init__(
        self,
        goal,
        cos_sim_threshold=0.6,
        similarity_modifier=0.001,
        temperature=0.6,
        top_p=0.9,
        rl_temp=0,
    ):
        super().__init__(
            goal, cos_sim_threshold, similarity_modifier, temperature, top_p, rl_temp
        )

    def give_values(self, observation):
        # this sets self.suggestions
        self.suggets(observation)

        # give cosine similarities for all possible actions over a certain threshold
        return np.array([self.compare(action, observation) for action in range(7)])


def obs_to_string(obs_matrix, positions=True, you=True):
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

            if not rel_x and not rel_y:
                observation_strings.append(
                    f"have a{state} {color} {item} in {'your' if you else 'my'} inventory"
                )
                continue

            string = f"see a{state} {color} {item}"

            if positions:
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
                    string += f" {longitude} and {latitude}"

                elif longitude and not latitude:
                    string += f" {longitude}"

                elif not longitude and latitude:
                    string += f" {latitude}"

            observation_strings.append(string)

    observation = f"{'You' if you else 'I'} {', '.join(observation_strings) if observation_strings else 'see nothing interesting'}."
    # Please only answer with a single of the following commands: RIGHT, LEFT, FORWARD or PICK UP."

    return observation


def seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_params(params_file):
    file = open(os.path.join(base_path,params_file))
    params = json.load(file)
    # print(json.dumps(params, indent=4, separators=(":", ",")))
    return params


def save_params(params):
    file = open(os.path.join(base_path, "hyperparams.json"), "w")
    json.dump(params, file, indent=4, separators=(",", ":"))
