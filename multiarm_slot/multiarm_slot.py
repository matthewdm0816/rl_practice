import random
from typing import List

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

def plot_reward_history(reward_history: List[float], name: str):
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(reward_history)
    
    fig.savefig(name, dpi=500)

if __name__ == "__main__":
    print(get_multiarm_slot().generate(7))