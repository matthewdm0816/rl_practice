from multiarm_slot import *
import random
from typing import List
import numpy as np

if __name__ == "__main__":

    multiarm_slot = get_multiarm_slot()
    N_STATES = 15
    N_ITER = 1000
    C = 0.5

    q = np.zeros(N_STATES)
    n = np.zeros(N_STATES)
    r_total = 0.
    r_history = []

    for _ in range(N_ITER):
        # random choose
        upper_bound = q + C * np.sqrt((np.log(n.sum()) / n))
        a, _ = argmax(upper_bound)

        r = multiarm_slot.generate(a)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]
        r_history.append(r)
        r_total += r

    plot_reward_history(r_history, "ucb")
    print(f"Estimated Q-values: {q}")
    print(f"Average Reward: {r_total / sum(n)}")
    
