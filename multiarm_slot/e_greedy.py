from multiarm_slot import get_multiarm_slot, argmax
import random
from typing import List


if __name__ == "__main__":

    multiarm_slot = get_multiarm_slot()

    N_STATES = 15
    N_ITER = 10000
    EPSILON = 0.1

    q = [0.] * N_STATES
    n = [0] * N_STATES  
    r_total = 0.
    for _ in range(N_ITER):
        # random choose
        if random.random() <= EPSILON:
            a = random.choice(range(N_STATES))
        else:
            a, _ = argmax(q)

        r = multiarm_slot.generate(a)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]
        r_total += r

    print(f"Estimated Q-values: {q}")
    print(f"Average Reward: {r_total / sum(n)}")
