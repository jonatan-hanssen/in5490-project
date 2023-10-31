import numpy as np
import matplotlib.pyplot as plt
import os, argparse, glob

plt.rcParams.update({"font.size": 26})
# plt.rcParams["font.family"] = "Times New Roman"
parser = argparse.ArgumentParser()

parser.add_argument("environment")
args = parser.parse_args()


data_path = os.path.join(os.path.dirname(__file__), "data/")

reward_data = list()
policy_data = list()
non_llm_data = list()

for file in os.listdir(data_path):
    if not file.endswith(".npy"):
        continue
    if not args.environment in file:
        continue

    data = np.load(os.path.join(data_path, file))

    if "llm" in file or "reward" in file:
        # print(file)
        # print(f"reward {np.mean(data)=}")
        reward_data.append(data)

    elif "policy" in file:
        print(f"policy {np.mean(data)=}")
        policy_data.append(data)

    else:
        # print(file)
        # print(f"none {np.mean(data)=}")
        non_llm_data.append(data)

reward_data = np.vstack(reward_data) if reward_data else np.zeros((1, 1991))
policy_data = np.vstack(policy_data) if policy_data else np.zeros((1, 1991))
non_llm_data = np.vstack(non_llm_data) if non_llm_data else np.zeros((1, 1991))

mean_reward_data = np.mean(reward_data, axis=0)
mean_policy_data = np.mean(policy_data, axis=0)
mean_non_llm_data = np.mean(non_llm_data, axis=0)

std_reward_data = np.std(reward_data, axis=0)
std_policy_data = np.std(policy_data, axis=0)
std_non_llm_data = np.std(non_llm_data, axis=0)

print(f"Increase reward: {np.mean(mean_reward_data) / np.mean(non_llm_data)}")
print(f"Increase policy: {np.mean(mean_policy_data) / np.mean(non_llm_data)}")


window_size = 15

means = list()
stds = list()
for window in np.array_split(mean_reward_data, len(mean_reward_data) // window_size):
    means.append(np.mean(window))

for window in np.array_split(std_reward_data, len(std_reward_data) // window_size):
    stds.append(np.mean(window))

x = np.arange(len(means)) * window_size
means = np.array(means)
stds = np.array(stds)
print(f"Standard deviation reward: {np.mean(stds)}")

plt.plot(x, means, label="LLM reward shaping", color="b", linewidth=3)
plt.fill_between(x, means - stds, means + stds, alpha=0.1, color="b", linewidth=4)

means = list()
stds = list()
for window in np.array_split(mean_non_llm_data, len(mean_non_llm_data) // window_size):
    means.append(np.mean(window))

for window in np.array_split(std_non_llm_data, len(std_non_llm_data) // window_size):
    stds.append(np.mean(window))
means = np.array(means)
stds = np.array(stds)
print(f"Standard deviation ppo: {np.mean(stds)}")

plt.plot(x, means, label="No LLM reward shaping", color="r", linewidth=3)
plt.fill_between(x, means - stds, means + stds, alpha=0.1, color="r", linewidth=4)

means = list()
stds = list()
for window in np.array_split(mean_policy_data, len(mean_policy_data) // window_size):
    means.append(np.mean(window))

for window in np.array_split(std_policy_data, len(std_policy_data) // window_size):
    stds.append(np.mean(window))
means = np.array(means)
stds = np.array(stds)
print(f"Standard deviation policy: {np.mean(stds)}")

plt.plot(x, means, label="LLM policy", color="g", linewidth=3)
plt.fill_between(x, means - stds, means + stds, alpha=0.1, color="g", linewidth=4)

plt.ylim(0, 1)
plt.title("Average reward per episode on DoorKey environment")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()

plt.show()
