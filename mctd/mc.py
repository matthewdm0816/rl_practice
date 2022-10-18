from utils import *
import copy
from matplotlib import pyplot as plt
import seaborn as sns
H, W = 4, 12
STEPS = 10_000

states = CliffStatesEpsGreedy(H, W)
states.init_values()
round_rewards = []
for step in range(STEPS):
    states.start_round()
    while True:
        try:
            states.random_walk()
        except StopIteration as e:
            break
        
    round_rewards.append(states.round_reward)

states.show_policy()

fig = plt.figure()
ax = fig.add_subplot(111)
clrs = sns.color_palette("husl", 2)
with sns.axes_style("darkgrid"):
    ax.plot(smoothed(round_rewards, gamma=0.8), c=clrs[0])

    ax.legend(["On-policy MC"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Round Reward")

fig.savefig("mc.png", dpi=500)
