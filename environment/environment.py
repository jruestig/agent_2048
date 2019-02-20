import numpy as np
if __name__ == "__main__":
    from movement import (shift_and_merge, action_to_dir_and_ax,
                          game_move)
else:
    from environment.movement import (shift_and_merge, action_to_dir_and_ax,
                                      game_move)


def start():
    state = np.zeros((4, 4), dtype=np.int32)
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


def step(state, reward, action, down_weight=1):
    state, r = game_move(state, *action_to_dir_and_ax(action))
    reward = reward + r
    return state, reward*down_weight
