import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("llm_file")
parser.add_argument("ppo_file")
args = parser.parse_args()

abs_path = os.path.dirname(__file__)

for arg in vars(args):
    file_name = os.path.join(abs_path, getattr(args, arg))

    data = np.load(file_name)

    means = list()
    for window in np.array_split(data, len(data) // 10):
        means.append(np.mean(window))

    plt.plot(
        range(len(means)), means, label=f"{getattr(args, arg)}".replace(".npy", "")
    )

plt.legend()
plt.show()
