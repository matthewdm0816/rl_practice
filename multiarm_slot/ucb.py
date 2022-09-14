from multiarm_slot import get_multiarm_slot, argmax
import random
from typing import List
import numpy as np

if __name__ == "__main__":

    multiarm_slot = get_multiarm_slot()

    N_STATES = 15
    N_ITER = 10000
    C = 0.5

    q = np.zeros(N_STATES)
    n = np.zeros(N_STATES)
    r_total = 0.

    for _ in range(N_ITER):
        # random choose
        upper_bound = a + C * (n.sum().log() / n).sqrt()
        a, _ = argmax(upper_bound)

        r = multiarm_slot.generate(a)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]
        r_total += r

    print(f"Estimated Q-values: {q}")
    print(f"Average Reward: {r_total / sum(n)}")
