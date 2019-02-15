import numpy as np


def start():
    state = np.zeros((4, 4))
    state[tuple(np.random.randint(0, 4, 2))] = np.random.choice((2,)*9 + (4,))
    score = 0
    return state, score
