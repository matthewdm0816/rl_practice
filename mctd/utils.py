import numpy as np
import copy
import random
from typing import Tuple


def tuple_add(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])


def compute_mean_std(rewards):
    # rewards: [N, T]
    mean = np.mean(rewards, axis=0)  # [T]
    # rewards_t = np.transpose(rewards) # [T, N]
    std = np.sqrt(np.sum((rewards - mean) ** 2, axis=0) / (rewards.shape[1] - 1))
    return mean, std


def window_average_ndim(xs, window_size=20):
    window_avg: np.ndarray = np.zeros_like(xs)
    zeros = np.zeros_like(xs)
    WINDOW_SIZE = window_size
    for k in range(1, WINDOW_SIZE + 1):
        window_avg += np.concatenate((xs[:, k:], zeros[:, :k]), axis=1)
    for k in range(-WINDOW_SIZE, 0):
        window_avg += np.concatenate((zeros[:, :-k], xs[:, :k]), axis=1)
    window_avg += xs
    window_avg /= (2.0 * WINDOW_SIZE) + 1
    return window_avg[:, WINDOW_SIZE:-WINDOW_SIZE]


def window_average(xs, window_size=20):
    return np.convolve(xs, np.ones(window_size), "valid") / window_size


def smoothed(xs: np.ndarray, gamma=0.8):
    smoothed = xs.copy()
    for i, x in zip(range(1, xs.shape[-1]), xs[1:]):
        smoothed[i] = gamma * x + smoothed[i - 1] * (1 - gamma)
    return smoothed


class CliffStatesv2:
    SHIFTS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    MAXSTEP = 100

    def __init__(self, h, w, gamma=0.9):
        self.h = h
        self.w = w
        self.gamma = 0.9
        self.terminal = (1, w)
        self.begin = (1, 1)
        # self.current_position = (1,1)

    def init_values(self):
        # initial values
        self.q_values = {
            (i, j, a): random.random()
            for a in range(4)
            for i in range(1, self.h + 1)
            for j in range(1, self.w + 1)
        }
        # self.avg_return = copy.deepcopy(self.q_values)
        self.return_counts = {
            (i, j, a): 0
            for a in range(4)
            for i in range(1, self.h + 1)
            for j in range(1, self.w + 1)
        }
        self.policy_values = {
            (i, j): 1 / 4 for i in range(1, self.h + 1) for j in range(1, self.w + 1)
        }
        # edge -> init 1/3
        for i in range(1, self.h + 1):
            self.policy_values[(i, 1)] = 1 / 3
            self.policy_values[(i, self.w)] = 1 / 3
        for j in range(1, self.w + 1):
            self.policy_values[(1, j)] = 1 / 3
            self.policy_values[(self.h, j)] = 1 / 3
        # corners -> 1/2
        for corner in [(1, 1), (1, self.h), (1, self.w), (self.h, self.w)]:
            self.policy_values[corner] = 1 / 2

    def reward(self, x, y):
        if x == 1:
            if y == self.w:
                return 0
            else:
                return -100

        return -1

    def sample_action(self, ok_mask, i: int, j: int, return_max=False):
        q_values = [self.q_values[(i, j, a)] for a in range(4)]
        maxa, maxval = -1, -1e7
        for a, val in enumerate(q_values):
            if ok_mask[a] and val > maxval:
                maxa = a
                maxval = val
        return maxa

    def get_ok_mask(self, x, y):
        ok_mask = [True] * 4
        if x == 1:
            ok_mask[0] = False
        if x == self.h:
            ok_mask[1] = False
        if y == 1:
            ok_mask[2] = False
        if y == self.w:
            ok_mask[3] = False
        return ok_mask

    def show_policy(self):
        arrows = ["↓", "↑", "←", "→"]
        for i in range(self.h, 0, -1):
            for j in range(1, self.w + 1):
                ok_mask = self.get_ok_mask(i, j)
                action = self.sample_action(ok_mask, i, j, return_max=True)
                print(arrows[action], end="")
            print("")

    def update_policy(self, r: float, x: int, y: int, a: int):
        last_state_action = (x, y, a)
        # self.avg_return[last_state_action] += self.round_reward
        self.return_counts[last_state_action] += 1
        self.q_values[last_state_action] += (
            r - self.q_values[last_state_action]
        ) / self.return_counts[last_state_action]

    def sample_route(self, until_end=False, begin=None):
        historical_action_state = set()
        if begin is not None:
            x, y = begin
        else:
            x, y = random.choice(range(1, self.h + 1)), random.choice(
                range(1, self.w + 1)
            )
        xs, ys, actions = [x], [y], []
        # total_steps = 0
        while True:
            # total_steps += 1
            # sample next action
            ok_mask = self.get_ok_mask(x, y)
            action = self.sample_action(ok_mask, x, y)
            action_state = (x, y, action)
            if until_end:
                # if total_steps > self.MAXSTEP:
                #     return [None] * 3
                if (x, y) == self.terminal:
                    break
            else:
                if action_state in historical_action_state:
                    break

            historical_action_state.add((x, y, action))
            x, y = tuple_add((x, y), self.SHIFTS[action])
            if x == 1 and 1 < y < self.w:
                x, y = 1, 1
            actions.append(action)
            xs.append(x)
            ys.append(y)

        return xs, ys, actions

    def random_walk(self) -> float:
        xs, ys, actions = self.sample_route()

        rewards = []
        reward = 0.0
        for x, y in zip(xs[::-1], ys[::-1]):
            reward = self.gamma * reward + self.reward(x, y)
            rewards.append(reward)

        rewards = rewards[::-1][1:]  # 0-th reward stands for nothing

        for x, y, a, r in zip(xs[:-1], ys[:-1], actions, rewards):
            self.update_policy(r, x, y, a)

        # return sum(rewards)
        return reward

    def calculate_reward(self) -> float:
        xs, ys, _ = self.sample_route(until_end=True, begin=self.begin)
        if xs is None:
            return -1e6
        rewards = []
        reward = 0.0
        for x, y in zip(xs[1:], ys[1:]):
            reward = self.gamma * reward + self.reward(x, y)
            rewards.append(reward)

        return sum(rewards)


class CliffStatesEpsGreedy(CliffStatesv2):
    def __init__(self, h, w, gamma=0.9, eps=0.05):
        super().__init__(h, w, gamma)
        self.eps = eps

    def sample_action(self, ok_mask, i, j, return_max=False):
        total_as = sum(ok_mask)
        q_values = [self.q_values[(i, j, a)] for a in range(4)]
        maxa, maxval = -1, -1e6
        for a, val in enumerate(q_values):
            if ok_mask[a] and val > maxval:
                maxa = a
                maxval = val
        if return_max:
            return maxa

        probs = [self.eps / total_as] * 4
        probs[maxa] += 1 - self.eps
        for a, ok in enumerate(ok_mask):
            if not ok:
                probs[a] = 0
        maxa = random.choices(range(4), weights=probs, k=1)[0]
        return maxa


class CliffStatesOffPolicyPolicyIterationEpsGreedy(CliffStatesv2):
    def __init__(self, h, w, gamma=0.9, eps=0.05):
        super().__init__(h, w, gamma)
        self.eps = eps

    def init_values(self):
        super().init_values()
        self.c_values = {
            (i, j, a): 0.0
            for a in range(4)
            for i in range(1, self.h + 1)
            for j in range(1, self.w + 1)
        }

    def sample_action(self, ok_mask, i, j, return_max=False):
        total_as = sum(ok_mask)
        q_values = [self.q_values[(i, j, a)] for a in range(4)]
        maxa, maxval = -1, -1e6
        for a, val in enumerate(q_values):
            if ok_mask[a] and val > maxval:
                maxa = a
                maxval = val
        if return_max:
            return maxa

        probs = [self.eps / total_as] * 4
        probs[maxa] += 1 - self.eps
        # probs = [1 / total_as] * 4
        for a, ok in enumerate(ok_mask):
            if not ok:
                probs[a] = 0
        maxa = random.choices(range(4), weights=probs, k=1)[0]
        p = probs[maxa]
        return maxa, p

    def sample_route(self, until_end=False, begin=None):
        historical_action_state = set()

        if begin is not None:
            x, y = begin
        else:
            x, y = random.choice(range(1, self.h + 1)), random.choice(
                range(1, self.w + 1)
            )
        xs, ys, actions, bs = [x], [y], [], []
        # total_steps = 0
        while True:
            # sample next action
            # total_steps += 1
            ok_mask = self.get_ok_mask(x, y)
            action, b = self.sample_action(ok_mask, x, y)
            action_state = (x, y, action)
            if until_end:
                # if total_steps > self.MAXSTEP:
                #     return [None] * 3
                if (x, y) == self.terminal:
                    break
            else:
                if action_state in historical_action_state:
                    break
            historical_action_state.add((x, y, action))
            x, y = tuple_add((x, y), self.SHIFTS[action])
            if x == 1 and 1 < y < self.w:
                x, y = 1, 1
            actions.append(action)
            xs.append(x)
            ys.append(y)
            bs.append(b)

        if begin is not None and until_end:
            return xs, ys, actions
        
        return xs, ys, actions, bs

    def update_policy(self, r: float, x: int, y: int, a: int, iw: float):
        last_state_action = (x, y, a)
        # self.avg_return[last_state_action] += self.round_reward
        self.c_values[last_state_action] += iw
        self.q_values[last_state_action] += (
            iw
            * (r - self.q_values[last_state_action])
            / self.c_values[last_state_action]
        )

    def random_walk(self) -> float:
        xs, ys, actions, bs = self.sample_route()

        rewards = []
        iws = []
        iw = 1.0
        reward = 0.0
        for x, y, b in zip(xs[1::-1], ys[1::-1], bs[::-1]):
            iws.append(iw)
            iw /= b
            reward = self.gamma * reward + self.reward(x, y)
            rewards.append(reward)

        rewards = rewards[::-1]

        for x, y, a, r, iw in zip(xs[:-1], ys[:-1], actions, rewards, iws):
            self.update_policy(r, x, y, a, iw)
            best_action = self.sample_action(
                self.get_ok_mask(x, y), x, y, return_max=True
            )
            if best_action != a:
                break

        # return sum(rewards)
        return reward


class CliffStatesTD0Sarsa(CliffStatesEpsGreedy):
    def __init__(self, h, w, gamma=0.9, eps=0.05, alpha=0.1):
        super().__init__(h, w, gamma, eps)
        self.alpha = alpha
        self.terminal = (1, w)

    def init_values(self):
        super().init_values()
        for a in range(4):
            self.q_values[(*self.terminal, a)] = 0

    def update_policy(
        self, r: float, x: int, y: int, a: int, next_state_action: Tuple[int, int, int]
    ):
        self.q_values[(x, y, a)] += self.alpha * (
            r + self.gamma * self.q_values[next_state_action] - self.q_values[(x, y, a)]
        )

    def sample_route(self, until_end=True, begin=None, update=False, return_rewards=False):
        xs, ys, actions, rewards = [], [], [], []

        if begin is not None:
            x, y = begin
        else:
            x, y = random.choice(range(1, self.h + 1)), random.choice(
                range(1, self.w + 1)
            )
        
        ok_mask = self.get_ok_mask(x, y)
        action = self.sample_action(ok_mask, x, y)
        # total_steps = 1
        while True:
            reward = self.reward(x, y)
            xs.append(x)
            ys.append(y)
            rewards.append(reward)
            # if total_steps > self.MAXSTEP:
            #     return [None] * 3
            if (x, y) == self.terminal:
                break
            # sample next action
            x_, y_ = tuple_add((x, y), self.SHIFTS[action])
            if x_ == 1 and 1 < y_ < self.w:
                x_, y_ = 1, 1
            ok_mask = self.get_ok_mask(x_, y_)
            action_ = self.sample_action(ok_mask, x_, y_)
            if update:
                self.update_policy(reward, x, y, action, (x_, y_, action_))

            x, y, action = x_, y_, action_
        
        if return_rewards:
            return xs, ys, actions, rewards
        else:
            return xs, ys, actions
    

    def random_walk(self) -> float:
        _, _, _, rewards = self.sample_route(update=True, return_rewards=True)
        r = 0
        for i in range(len(rewards)-1,-1,-1):
            rewards[i] = self.gamma * rewards[i] + r 
            r = rewards[i]
        return rewards[0]

class CliffStatesTD0ExpectedSarsa(CliffStatesTD0Sarsa):
    def __init__(self, h, w, gamma=0.9, eps_policy=1, alpha=0.1, eps_sampler=0.05):
        super().__init__(h, w, gamma, eps_sampler, alpha)
        self.eps_policy = eps_policy

    def update_policy(
        self, r: float, x: int, y: int, a: int, next_state_action: Tuple[int, int, int]
    ):
        x_, y_, a_ = next_state_action
        ok_mask = self.get_ok_mask(x_, y_)
        total_as = sum(ok_mask)
        q_values = np.array([self.q_values[(x_, y_, a)] for a in range(4)])

        maxa = np.argmax(q_values)
        # assert a == maxa

        probs = np.ones(4) * (self.eps_policy / total_as)
        probs[maxa] += 1 - self.eps_policy
        probs[ok_mask] = 0
        # print(probs)
        expected_q = float(np.sum(q_values * probs))
        print(expected_q)
        # expected_q = self.q_values[next_state_action]
        self.q_values[(x, y, a)] += self.alpha * (
            r + self.gamma * expected_q - self.q_values[(x, y, a)]
        )

def run_exp(args):
    rw_class, H, W, STEPS = args
    states = rw_class(H, W)
    states.init_values()
    round_rewards = []
    for step in range(STEPS):
        rw = states.random_walk()
        round_rewards.append(rw)

    states.show_policy()
    print("-" * 20)
    return round_rewards


