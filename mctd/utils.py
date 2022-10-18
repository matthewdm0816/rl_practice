import numpy as np
import copy
import random

def tuple_add(t1, t2):
    return (t1[0]+t2[0],t1[1]+t2[1])

def smoothed(xs, gamma=0.8):
    smoothed = xs[:1]
    for x in xs[1:]:
        smoothed.append(gamma * x + smoothed[-1] * (1-gamma))
    return smoothed

class CliffStates:
    SHIFTS = ((-1,0),(1,0),(0,-1),(0,1))

    def __init__(self, h, w, gamma=0.9):
        self.h = h
        self.w = w
        self.gamma = 0.9
        # self.current_position = (1,1)

    def init_values(self):
        # initial values
        self.q_values = {(i,j,a): random.random() for a in range(4) for i in range(1, self.h+1) for j in range(1, self.w+1)}
        # self.avg_return = copy.deepcopy(self.q_values)
        self.return_counts = {(i,j,a): 0 for a in range(4) for i in range(1, self.h+1) for j in range(1, self.w+1)}
        self.policy_values = {(i,j): 1/4 for i in range(1,self.h+1) for j in range(1,self.w+1)}
        # edge -> init 1/3
        for i in range(1,self.h+1):
            self.policy_values[(i,1)] = 1/3
            self.policy_values[(i,self.w)] = 1/3
        for j in range(1,self.w+1):
            self.policy_values[(1,j)]=1/3
            self.policy_values[(self.h,j)]=1/3
        # corners -> 1/2
        for corner in [(1,1),(1,self.h),(1,self.w),(self.h,self.w)]:
            self.policy_values[corner] = 1/2

    def reward(self, x, y):
        if x == 1:
            if y == self.w:
                return 0
            else:
                return -100

        return -1

    def sample_action(self, ok_mask, i=None, j=None):
        if i is None:
            i, j = self.current_position
        q_values = [self.q_values[(i, j, a)] for a in range(4)]
        maxa, maxval = -1, -1e7
        for a, val in enumerate(q_values):
            if ok_mask[a] and val > maxval:
                maxa = a
                maxval = val
        return maxa

    
    def show_policy(self):
        arrows = ["↓","↑","←","→"]
        for i in range(self.h,0,-1):
            for j in range(1,self.w+1):
                ok_mask = [True] * 4
                if i == 1:
                    ok_mask[0] = False
                if i == self.h:
                    ok_mask[1] = False
                if j == 1:
                    ok_mask[2] = False
                if j == self.w:
                    ok_mask[3] = False
                action = self.sample_action(ok_mask, i, j)
                print(arrows[action],end="")
            print("")

    def update_policy(self):
        last_state_action = (*self.last_state, self.last_action)
        # self.avg_return[last_state_action] += self.round_reward
        self.return_counts[last_state_action] += 1
        self.q_values[last_state_action] += (self.round_reward - self.q_values[last_state_action]) / self.return_counts[last_state_action]

    def random_walk(self):
        x,y = self.current_position

        # update round reward
        reward = self.reward(x,y)
        self.round_reward = self.gamma * self.round_reward + reward
        if self.last_state is not None:
            if (self.last_state, self.last_action) in self.historical_action_state:
                raise StopIteration
            self.historical_action_state.add((self.last_state, self.last_action))
        
            # update q values
            self.update_policy()

        # sample next action
        ok_mask = [True] * 4
        if x == 1:
            ok_mask[0] = False
        if x == self.h:
            ok_mask[1] = False
        if y == 1:
            ok_mask[2] = False
        if y == self.w:
            ok_mask[3] = False
        action = self.sample_action(ok_mask)
        self.last_action = action
        self.last_state = self.current_position

        self.current_position = tuple_add(self.current_position, self.SHIFTS[action])

    def start_round(self):
        # self.current_position = (random.choice(range(1,self.h+1)), random.choice(range(1,self.w+1)))
        self.current_position = (1,1)
        self.last_state, self.last_action = None, None
        self.round_reward = 0
        self.historical_action_state = set()

class CliffStatesEpsGreedy(CliffStates):
    def __init__(self, h, w, gamma=0.9, eps=0.05):
        super().__init__(h,w,gamma)
        self.eps = eps
    def sample_action(self, ok_mask, i=None, j=None, return_max=False):
        # TODO
        total_as = sum(ok_mask)
        if i is None:
            i, j = self.current_position
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
                probs[a]=0
        maxa = random.choices(range(4),weights=probs, k=1)[0]
        return maxa