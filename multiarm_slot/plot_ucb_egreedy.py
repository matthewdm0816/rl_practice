import json
from multiarm_slot import *
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

ucb_data = json.load(open("ucb.json", "r"))
egreedy_data = json.load(open("e_greedy.json", "r"))

ucb_data = np.array(ucb_data, dtype=np.float32)
egreedy_data = np.array(egreedy_data, dtype=np.float32)

ucb_data = window_average_ndim(ucb_data)
egreedy_data = window_average_ndim(egreedy_data)

ucb_data, ucb_std = compute_mean_std(ucb_data)
egreedy_data, egreedy_std = compute_mean_std(egreedy_data)

fig = plt.figure()
ax = fig.add_subplot(111)

# ax.plot(window_average(ucb_data))
# ax.plot(window_average(egreedy_data))
clrs = sns.color_palette("husl", 2)
with sns.axes_style("darkgrid"):
    ax.plot(ucb_data, c=clrs[0])
    ax.fill_between(np.arange(ucb_data.shape[0]), ucb_data + ucb_std, ucb_data - ucb_std, alpha=0.3, facecolor=clrs[0])
    ax.plot(egreedy_data, c=clrs[1])
    ax.fill_between(np.arange(ucb_data.shape[0]), egreedy_data + egreedy_std, egreedy_data - egreedy_std, alpha=0.3, facecolor=clrs[1])
    # ax.legend()
    # ax.set_yscale("log")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Iteration")
    ax.legend(["UCB-mean", "UCB-std", "$\\varepsilon$-greedy-mean", "$\\varepsilon$-greedy-std"])

fig.savefig("compare.png", dpi=500)