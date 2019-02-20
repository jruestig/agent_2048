import numpy as np
from environment.movement import game_move, action_to_dir_and_ax
from environment.environment import start, step, legal_moves


def monte_carlo_tree_value(fields, rewards, maxmoves=20, returnscores=False,
                           startaction=None, down_weight=1):
    n = len(rewards)
    score = np.zeros(n)

    if startaction is not None:
        fields, rewards = game_move(fields, *action_to_dir_and_ax(startaction))
        score += rewards * down_weight

    for i in range(0, maxmoves):
        actions = np.random.randint(0, 4, n)

        for a in (0, 1, 2, 3):  # *action_to_dir_and_ax(a)
            fields[actions == a], rewards[actions == a] = (
                game_move(fields[actions == a], *action_to_dir_and_ax(a)))

        score += rewards * down_weight

    if returnscores:
        return score
    else:
        return np.mean(score)


class uctree():
    def __init__(self, depth, breadth, rolloutlength):
        self.depth = int(depth)
        self.breadth = int(breadth)
        self.tree = dict()
        self.rolloutlen = rolloutlength
        self.ucb_explore = 0.5
        self.down_weight = 0.4

    def act(self, s):
        legal = legal_moves(s)
        q, n = self.predict(s)
        # qn = q/n
        qn = np.zeros(4)
        qn[legal] = (q/n)[legal]
        a = np.argmax(qn)
        if q.sum() == 0:
            print(s)
            assert False
        self.tree = dict()
        return a

    def predict(self, roots):
        # returns the ucb for root state
        for ii in range(self.depth):
            a_s = ("root",)
            s, r = self.expand(roots, self.breadth)
            s, r, a_s = self.selection_expansion(s, r, a_s)
            r = self.evaluation(s, r)
            self.backup(r, a_s)
        return self.tree[("root",)]

    def selection_expansion(self, s, r, a_s):
        # selects leaf node with highest ucb
        if a_s not in self.tree:
            self.tree[a_s] = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]
            return s, r, a_s
        q, n = self.tree[a_s][0], self.tree[a_s][1]
        a = self.ucb1(q, n)
        a_s = a_s + (a,)
        s, r = step(s, r, a_s[-1], self.down_weight)
        return self.selection_expansion(s, r, a_s)

    def evaluation(self, s, r):
        # returns rewards of each simulation
        q = np.zeros(4)
        for a in (0, 1, 2, 3):
            q[a] = monte_carlo_tree_value(s, r, self.rolloutlen,
                                          down_weight=self.down_weight,
                                          returnscores=True,
                                          startaction=a).sum()
        return q

    def backup(self, q, actions):
        self.tree[actions][0] = q
        self.tree[actions][1] = np.array((self.breadth,)*4)
        r = q.sum()
        actions = list(actions)
        for ii in range(len(actions)-1):
            a = actions.pop(-1)
            path = tuple(actions)
            self.tree[path][0][a] += r
            self.tree[path][1][a] += self.breadth * 4

    def ucb1(self, q, n):
        # returns argmax according to ucb
        q = self.ucb_explore * q/n  # np.where(q > 0, q/n, 10e9)
        u = np.sqrt(np.log(n.sum()/(np.where(n > 0, n, 1))))
        return np.argmax(q+u)

    def expand(self, s, n):
        r = np.zeros(n)
        s = np.ones((n,) + s.shape) * s
        return s, r


agent = uctree(depth=1, breadth=1000, rolloutlength=4)
state, score = start()

for ii in range(3000):
    # print(state)
    a = agent.act(state)
    state, score = step(state, score, a)
    # print("-"*40)
