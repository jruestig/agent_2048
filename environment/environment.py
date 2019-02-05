import numpy as np
from random import choice
if __name__ == "__main__":
    from movement import move
else:
    from .movement import move


class Environment():
    def start(self):
        board = self.place_two(np.zeros([4, 4], np.int64))
        score = 0
        done = False
        return board, score, done

    def step(self, board, score, action):
        nboard, score = move(board, score, action)
        nboard = self.place_two(nboard)
        done = self.check_done(nboard)
        return nboard, score, done

    def check_done(self, board):
        legal_action = self.legal_actions(board)
        if legal_action == []:
            return True
        return False

    @staticmethod
    def legal_actions(board):
        legal_action = []
        for action in range(4):
            n_board, _ = move(board, 0, action)
            if not((n_board-board == np.full_like(board, 0)).all()):
                legal_action.append(action)
        return legal_action

    @staticmethod
    def illegal_actions(board):
        illegal_action = []
        for action in range(4):
            n_board, _ = move(board, 0, action)
            if ((n_board-board == np.full_like(board, 0)).all()):
                illegal_action.append(action)
        return illegal_action

    @staticmethod
    def place_two(oboard):
        board = oboard.copy()
        if (board == 0).any():
            a = 2
            while a != 0:
                cr = np.random.randint(0, board.shape[0])
                rr = np.random.randint(0, board.shape[1])
                a = board[cr][rr]
            board[cr][rr] = choice([2, 2, 2, 2, 2, 2, 2, 2, 2, 4])
        return board


if __name__ == "__main__":
    bo = Environment()
    board, score, done = bo.start()
    for ii in range(200):
        print(40*"-")
        action = choice(bo.legal_actions(board))
        if action == 0:
            print("left")
        elif action == 1:
            print("right")
        elif action == 2:
            print("up")
        elif action == 3:
            print("down")
        board, score, done = bo.step(board, score, action)
        print(len(bo.legal_actions(board)))
        print(board)
        if done:
            break
