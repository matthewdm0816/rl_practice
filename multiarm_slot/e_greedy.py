from multiarm_slot import *
import random
from typing import List
import json

if __name__ == "__main__":

    multiarm_slot = get_multiarm_slot()

    N_STATES = 15
    N_ITER = 1000
    EPSILON = 0.05

    r_histories = []
    for _ in range(100):
        q = [0.] * N_STATES
        n = [0] * N_STATES  
        r_total = 0.
        r_history = []

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
            r_history.append(r)

        print(f"Estimated Q-values: {q}")
        print(f"Average Reward: {r_total / sum(n)}")
        r_histories.append(r_history)
    
    json.dump(r_histories, open("e_greedy.json", "w"))