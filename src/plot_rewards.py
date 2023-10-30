import numpy as np
import matplotlib.pyplot as plt
import os, argparse, glob

plt.rcParams.update({"font.size": 26})
# plt.rcParams["font.family"] = "Times New Roman"
parser = argparse.ArgumentParser()

parser.add_argument("environment")
args = parser.parse_args()


data_path = os.path.join(os.path.dirname(__file__), "data/")

llm_data = list()
non_llm_data = list()

for file in os.listdir(data_path):
    if not file.endswith(".npy"):
        continue
    if not args.environment in file:
        continue

    data = np.load(os.path.join(data_path, file))

    if "llm" in file:
        llm_data.append(data)

    else:
        non_llm_data.append(data)

llm_data = np.vstack(llm_data)
non_llm_data = np.vstack(non_llm_data)

mean_llm_data = np.mean(llm_data, axis=0)
mean_non_llm_data = np.mean(non_llm_data, axis=0)

std_llm_data = np.std(llm_data, axis=0)
std_non_llm_data = np.std(non_llm_data, axis=0)

print(np.mean(mean_llm_data) / np.mean(non_llm_data))
print(f"{np.mean(std_llm_data)=}")
print(f"{np.mean(std_non_llm_data)=}")


window_size = 15

means = list()
stds = list()
for window in np.array_split(mean_llm_data, len(mean_llm_data) // window_size):
    means.append(np.mean(window))

for window in np.array_split(std_llm_data, len(std_llm_data) // window_size):
    stds.append(np.mean(window))

x = np.arange(len(means)) * window_size
means = np.array(means)
stds = np.array(stds)
print(np.mean(stds))

line_llm = plt.plot(x, means, label="LLM reward shaping", color="b", linewidth=3)
fill_llm = plt.fill_between(
    x, means - stds, means + stds, alpha=0.1, color="b", linewidth=4
)

means = list()
stds = list()
for window in np.array_split(mean_non_llm_data, len(mean_non_llm_data) // window_size):
    means.append(np.mean(window))

for window in np.array_split(std_non_llm_data, len(std_non_llm_data) // window_size):
    stds.append(np.mean(window))
means = np.array(means)
stds = np.array(stds)
print(np.mean(stds))

line_non_llm = plt.plot(x, means, label="No LLM reward shaping", color="r", linewidth=3)
fill_non_llm = plt.fill_between(
    x, means - stds, means + stds, alpha=0.1, color="r", linewidth=4
)

plt.title("Average reward per episode on DoorKey environment")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()

plt.show()
