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

if __name__ == "__main__":
    print(get_multiarm_slot().generate(7))