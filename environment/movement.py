import numpy as np


def place_random(oboard):
    board = oboard.copy()
    a = 2
    count = board.shape[0]*board.shape[1]
    while a != 0:
        cr = np.random.randint(0, board.shape[0])
        rr = np.random.randint(0, board.shape[1])
        a = board[cr][rr]
        count -= 1
        if count < 0:
            return board
    board[cr][rr] = 2
    return board


def merge_left(oboard, score):
    def shift_zero(array, times):
        for k in range(times):
            for ii in range(len(array)-1):
                if array[ii] == 0:
                    array[ii] = array[ii+1]
                    array[ii+1] = 0
    board = oboard.copy()
    for row in board:
        count = 0
        shift_zero(row, (row == 0).sum())
        for ii in range(len(row)-1):
            if row[ii] == row[ii+1]:
                row[ii] *= 2
                score += int(row[ii])
                row[ii+1] = 0
                count += 1
        shift_zero(row, count)
    return board, score


def merge_right(board, score):
    board = reverse(board)
    board, score = merge_left(board, score)
    return reverse(board), score


def merge_up(board, score):
    board = board.T
    board, score = merge_left(board, score)
    return board.T, score


def merge_down(board, score):
    board = board.T
    board, score = merge_right(board, score)
    return board.T, score


def reverse(board):
    new = []
    for i in range(len(board)):
        new.append([])
        for j in range(len(board[0])):
            new[i].append(board[i][len(board[0])-j-1])
    return np.array(new)


def move(board, score, direction):
    oboard = board.copy()
    if direction == 0:
        board, score = merge_left(oboard, score)
    elif direction == 1:
        board, score = merge_right(oboard, score)
    elif direction == 2:
        board, score = merge_up(oboard, score)
    elif direction == 3:
        board, score = merge_down(oboard, score)
    return board, score


def game_state(mat):
    if (mat == 0).any():
        return "not over"
    for i in range(len(mat)-1):  # intentionally reduced to check the row on the right and below
        for j in range(len(mat[0])-1):  # more elegant to use exceptions but most likely this will be their solution
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    for k in range(len(mat)-1):  # to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):   # check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'


def done(mat):
    # Not working
    if (mat == 0).any():
        return "not over"
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            print(i, j)
            try:
                if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                    return 'not over'
            except:
                try:
                    if mat[len(mat)-1][j] == mat[len(mat)-1][j+1]:
                        return 'not over'
                except:
                    if mat[i][len(mat)-1] == mat[i+1][len(mat)-1]:
                        return 'not over'
                else:
                    return 'lose'

if __name__ == "__main__":
    board = np.zeros((3, 3))
    score = 0
    for ii in range(162):
        board = place_random(board)
        print(40*"-")
        board, score = move(board, score, np.random.randint(0, 4))
        print(board)
        if (game_state(board) != done(board)):
            print(game_state(board), done(board))
        if game_state(board) == "lose":
            break
