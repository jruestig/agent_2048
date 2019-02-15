import numpy as np


def shift_and_merge(arr, direction=-1, axis=-1):
    # returns narr, reward
    def shift(arr, direction=1, axis=-1):
        if direction < 0:
            seg = (arr == 0)
        else:
            seg = arr > 0

        srt = np.argsort(seg, axis=axis)

        return np.take_along_axis(arr, srt, axis=axis)

    def slice_axis(axis, myslice):
        if axis == -1:
            return np.s_[..., myslice]
        elif axis == -2:
            return np.s_[..., myslice, :]
        else:
            assert 0, "wrong :P"

    arrn = shift(arr, axis=axis, direction=direction)

    if direction < 0:
        slice_p1 = slice_axis(axis, np.s_[1:])
        slice_0 = slice_axis(axis, np.s_[:-1])
    else:
        slice_0 = slice_axis(axis, np.s_[1:])
        slice_p1 = slice_axis(axis, np.s_[:-1])

    havetomerge = (arrn[slice_p1] == arrn[slice_0]) & (arrn[slice_0] > 0)

    if direction < 0:
        inds = range(0, havetomerge.shape[axis]-1)
    else:
        inds = range(havetomerge.shape[axis]-1, 0, -1)

    for i in inds:
        if direction < 0:
            slice_ip1 = slice_axis(axis, i+1)
            slice_i0 = slice_axis(axis, i)
        else:
            slice_i0 = slice_axis(axis, i)
            slice_ip1 = slice_axis(axis, i-1)

        havetomerge[slice_ip1] = (havetomerge[slice_ip1] &
                                  (~havetomerge[slice_i0]))

    reward = 2 * np.sum(havetomerge * arrn[slice_0], axis=(-1, -2))

    arrn[slice_0][havetomerge] *= 2
    arrn[slice_p1][havetomerge] = 0

    arrn = shift(arrn, direction=direction, axis=axis)

    return arrn, reward


def game_move(arr, direction=-1, axis=-1):
    arrn, reward = shift_and_merge(arr, direction=direction, axis=axis)
    nothing_changed = np.min(arr == arrn, axis=(-1, -2)).flatten()

    arrnflat = arrn.reshape((-1, ) + arrn.shape[-2:])

    num_zero = np.sum(arrnflat == 0, axis=(-1, -2))
    newpos = np.random.randint(0, 13837458124, arrnflat.shape[0])  # % num_zero

    newpos = newpos % np.clip(num_zero, 1, None)
    delete = np.where((num_zero == 0) | nothing_changed)

    newpos = np.delete(newpos, delete)
    num_zero = np.delete(num_zero, delete)

    idzero = np.nonzero(arrnflat == 0)

    newpos_shift = newpos + (np.cumsum(num_zero) - num_zero)

    idselect = [idzero[0][newpos_shift], idzero[1][newpos_shift],
                idzero[2][newpos_shift]]

    arrnflat[tuple(idselect)] = np.random.choice((2,)*9 + (4,),
                                                 size=(len(newpos)))
    return arrnflat.reshape(arr.shape), reward


def action_to_dir_and_ax(action):
    if action == 0:
        return -1, -1
    elif action == 1:
        return -1, -2
    elif action == 2:
        return 1, -1
    elif action == 3:
        return 1, -2
    else:
        assert 0, action
