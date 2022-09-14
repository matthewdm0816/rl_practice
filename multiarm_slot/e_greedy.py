from multiarm_slot import get_multiarm_slot
import random
from typing import List

def argmax(values: List[float]):
    max_val, max_ind = -1e10, 0
    for idx, val in enumerate(values):
        if val > max_val:
            max_val, max_ind = val, idx
    return max_ind, max_val

if __name__ == "__main__":

    multiarm_slot = get_multiarm_slot()

    N_STATES = 15
    N_ITER = 10000
    EPSILON = 0.1

    q = [0.] * N_STATES
    n = [0] * N_STATES

    for _ in range(N_ITER):
        # random choose
        if random.random() <= EPSILON:
            a = random.choice(range(N_STATES))
        else:
            a, _ = argmax(q)

        r = multiarm_slot.generate(a)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]

    print(f"Estimated Q-values: {q}")
