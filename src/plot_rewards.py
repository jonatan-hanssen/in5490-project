import numpy as np
import matplotlib.pyplot as plt
import os, argparse, glob

plt.rcParams.update({"font.size": 22})
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

llm_data = np.mean(np.vstack(llm_data), axis=0)
non_llm_data = np.mean(np.vstack(non_llm_data), axis=0)

window_size = 15


means = list()
for window in np.array_split(llm_data, len(llm_data) // window_size):
    means.append(np.mean(window))

plt.plot(np.arange(len(means)) * window_size, means, label="LLM reward shaping")

means = list()
for window in np.array_split(non_llm_data, len(non_llm_data) // window_size):
    means.append(np.mean(window))

plt.plot(np.arange(len(means)) * window_size, means, label="No LLM reward shaping")

plt.title("Average reward per episode on DoorKey environment")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.show()
