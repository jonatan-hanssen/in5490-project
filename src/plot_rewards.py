import numpy as np
import matplotlib.pyplot as plt
import os, argparse, glob

plt.rcParams.update({"font.size": 26})
# plt.rcParams["font.family"] = "Times New Roman"
parser = argparse.ArgumentParser()

parser.add_argument("environment")
parser.add_argument("-w", "--window", type=int, default=20)
args = parser.parse_args()


data_path = os.path.join(os.path.dirname(__file__), "data/")

reward_data = list()
policy_data = list()
non_llm_data = list()
both_data = list()

for file in os.listdir(data_path):
    if not file.endswith(".npy"):
        continue
    if not args.environment.lower() in file:
        continue

    data = np.load(os.path.join(data_path, file))
    print(f"{file=}")
    print(f"{len(data)=}")

    if "llm" in file or "reward" in file:
        reward_data.append(data)

    elif "policy" in file:
        policy_data.append(data)

    elif "both" in file:
        both_data.append(data)

    elif len(file) < len(args.environment) + 6:
        non_llm_data.append(data)
    print(f"{len(file)=}")
    print(f"{len(args.environment)=}")

DATA_SIZE = 191 if args.environment.lower() == "empty" else 1991

data_list = [reward_data, policy_data, both_data, non_llm_data]
color_list = ["r", "g", "k", "b"]
title_list = ["LLM reward shaping", "LLM policy influencing", "Both", "Baseline"]

means_list = list()

for i in range(4):
    print(f"Now calculating for: {title_list[i]}")
    data = np.vstack(data_list[i]) if data_list[i] else np.zeros((1, DATA_SIZE))
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    rolling_mean = np.convolve(
        np.pad(mean, args.window, mode="symmetric"),
        np.ones(args.window) / args.window,
        mode="same",
    )[args.window : -args.window]

    x = np.arange(len(rolling_mean))
    means = np.array(mean)
    print(f"Standard deviation: {np.mean(std)}")
    print(f"Mean: {np.mean(mean)}")
    print()
    print("-" * 60)
    print()

    means_list.append(np.mean(means))

    plt.plot(
        x[:: args.window],
        rolling_mean[:: args.window],
        label=title_list[i],
        color=color_list[i],
        linewidth=3,
    )
    # plt.fill_between(
    #     x, means - stds, means + stds, alpha=0.1, color=color_list[i], linewidth=4
    # )

print(f"Increase reward shaping: {means_list[0] / means_list[3]}")
print(f"Increase policy: {means_list[1] / means_list[3]}")
print(f"Increase both: {means_list[2] / means_list[3]}")

plt.ylim(0, 1)
plt.title(f"Average reward per episode\n on {args.environment} environment")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()

plt.show()
