import json
from multiarm_slot import window_average
from matplotlib import pyplot as plt
    
ucb_data = json.load(open("ucb.json", "r"))
egreedy_data = json.load(open("e_greedy.json", "r"))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(window_average(ucb_data))
ax.plot(window_average(egreedy_data))

ax.legend(["UCB", "$\\varepsilon$-greddy"])

fig.savefig("compare.png", dpi=500)