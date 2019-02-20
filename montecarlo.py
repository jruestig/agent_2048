import numpy as np
from environment.movement import game_move, action_to_dir_and_ax
from environment.environment import start


def monte_carlo_tree_value(field, n, maxmoves=20, returnscores=False,
                           startaction=None):
    score = np.zeros(n)
    rewards = np.zeros(n)
    fields = np.ones((n,) + field.shape) * field

    if startaction is not None:
        fields, rewards = game_move(fields, *action_to_dir_and_ax(startaction))
        score += rewards

    for i in range(0, maxmoves):
        actions = np.random.randint(0, 4, n)

        for a in (0, 1, 2, 3):  # *action_to_dir_and_ax(a)
            fields[actions == a], rewards[actions == a] = (
                game_move(fields[actions == a], *action_to_dir_and_ax(a)))

        score += rewards

    if returnscores:
        return score
    else:
        return np.mean(score)


def monte_carlo_tree_action_values(field, n, maxmoves=20):
    q = np.zeros(4)
    for a in (0, 1, 2, 3):
        q[a] = monte_carlo_tree_value(field, n, maxmoves=maxmoves,
                                      startaction=a)
    return q


class Monte():
    def __init__(self, breadth, depth):
        self.breadth = breadth
        self.depth = depth

    def act(self, state):
        q = monte_carlo_tree_action_values(state, self.breadth, self.depth)
        return q, np.argmax(q)


if __name__ == "__main__":
    agent = Monte(160, 20)
    state, score = start()

    for ii in range(1000000):
        q, a = agent.act(state)
        state, sc = game_move(state, *action_to_dir_and_ax(a))
        score += sc
        if (q == np.zeros([4])).all():
            print(state)
            break
        if ii % 3:
            print(state, q)
