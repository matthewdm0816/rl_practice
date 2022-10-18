import random
from typing import List, Tuple
import numpy as np
import json

WORLD_SIZE = 5
POS_A = (0, 1)
POS_AA = (4, 1)
POS_B = (0, 3)
POS_BB = (2, 3)
SHIFT = [(0, 1), (1, 0), (0, -1), (-1, 0)]
GAMMA = 0.9

def tuple_add(t1: Tuple, t2: Tuple) -> Tuple:
    return tuple(x + y for x, y in zip(t1, t2))

def inbound(pos: Tuple[int, int]) -> bool:
    return WORLD_SIZE > pos[0] >= 0 and WORLD_SIZE > pos[1] >= 0


def rw(n_round: int):
    # random walk on board, for a max of *n_round* steps
    value = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.float64)
    old_value = value.copy()
    n_value = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.int32) + 1
    current_pos = (0, 0)
    for idx in range(n_round):
        shift = random.choice(SHIFT)
        next_pos = tuple_add(current_pos, shift)
        if not inbound(next_pos):
            reward = -1
            next_pos = current_pos
        elif current_pos == POS_A:
            reward = 10
            next_pos = POS_AA
        elif current_pos == POS_B:
            reward = 5
            next_pos = POS_BB
        else:
            reward = 0

        # value[current_pos] *= (n_value[current_pos] - 1) / n_value[current_pos]
        value[current_pos] += (reward + GAMMA * value[next_pos] - value[current_pos]) / n_value[current_pos]
        n_value[current_pos] += 1

        current_pos = next_pos

        if (idx + 1) % 1000 == 0:
            diff = np.max(np.abs(value - old_value))
            if diff <= 1e-4:
                print(f"Meet convergence criteria @ {idx}-step")
                break
            old_value = value.copy()

    return value


def rw_mean(n_epochs: int):
    results = [rw(10_000_000) for _ in range(n_epochs)]
    result = sum(results[1:], start=results[0])
    return result / len(results)
    

if __name__ == "__main__":
    print(rw(10_000_000))
    # print(rw(10000))
    print(rw_mean(10))