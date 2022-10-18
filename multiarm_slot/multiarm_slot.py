import random
from typing import List
import numpy as np

class MultiArmSlot:
    def __init__(self, means: List[float], stds: List[float]):
        self.means = means
        self.stds = stds

    def generate(self, idx):
        return random.gauss(self.means[idx], self.stds[idx])

def get_multiarm_slot():
    return MultiArmSlot(list(range(1, 16)), stds=[1] * 15)

def argmax(values: List[float]):
    max_val, max_ind = -1e10, 0
    for idx, val in enumerate(values):
        if val > max_val:
            max_val, max_ind = val, idx
    return max_ind, max_val

def window_average(xs):
    window_avg: np.ndarray = np.zeros_like(xs)
    WINDOW_SIZE = 5
    for k in range(1, WINDOW_SIZE + 1):
        window_avg += np.concatenate((xs[k:], np.zeros(k)))
    for k in range(-WINDOW_SIZE, 0):
        window_avg += np.concatenate((np.zeros(-k), xs[:k]))
    window_avg += xs
    window_avg /= (2. * WINDOW_SIZE) + 1
    return window_avg[:-WINDOW_SIZE]

def plot_reward_history(reward_history: List[float], name: str):
    import json
    json.dump(reward_history, open(f"{name}.json", "w"))

    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    window_avg = window_average(reward_history)
    ax.plot(window_avg)
    
    fig.savefig(name, dpi=500)

def window_average_ndim(xs):
    window_avg: np.ndarray = np.zeros_like(xs)
    zeros = np.zeros_like(xs)
    WINDOW_SIZE = 2
    for k in range(1, WINDOW_SIZE + 1):
        window_avg += np.concatenate((xs[:,k:], zeros[:,:k]), axis=1)
    for k in range(-WINDOW_SIZE, 0):
        window_avg += np.concatenate((zeros[:,:-k], xs[:, :k]), axis=1)
    window_avg += xs
    window_avg /= (2. * WINDOW_SIZE) + 1
    return window_avg[:, :-WINDOW_SIZE]
    
def compute_mean_std(rewards):
    # rewards: [N, T]
    mean = np.mean(rewards, axis=0) # [T]
    # rewards_t = np.transpose(rewards) # [T, N]
    std = np.sum((rewards - mean) ** 2, axis=0) / (rewards.shape[1] - 1)
    return mean, std

if __name__ == "__main__":
    print(get_multiarm_slot().generate(7))