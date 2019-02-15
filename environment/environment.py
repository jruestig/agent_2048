import numpy as np
from environment.movement import shift_and_merge, action_to_dir_and_ax


def start():
    state = np.zeros((4, 4))
    state[tuple(np.random.randint(0, 4, 2))] = np.random.choice((2,)*9 + (4,))
    score = 0
    return state, score


def legal_moves(state):
    lmoves = np.empty(state.shape[:-2] + (4,), np.bool)
    for i in range(4):
        nstate, _ = shift_and_merge(state, *action_to_dir_and_ax(i))
        nothing_changed = np.min(nstate == state, axis=(-1, -2)).flatten()
        lmoves[..., i] = not nothing_changed
    return lmoves
