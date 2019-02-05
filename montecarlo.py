import numpy as np
from random import choice
from environment.environment import Environment


class MonteCarlo():
    def __init__(self, sample_size=40):
        self.sample_size = int(sample_size/4)

    def roll_out(self, env, board):
        moves = []
        for action in range(4):
            moves.append(self.run(action, env, board))
        return moves

    def run(self, action, env, board):
        if action not in env.legal_actions(board):
            return 0.
        cboards = [board for _ in range(self.sample_size)]
        runs = []
        for board in cboards:
            board, score, done = env.step(board, 0, action)
            while not done:
                move = choice(env.legal_actions(board))
                board, score, done = env.step(board, score, move)
            runs.append(score)
        return np.array(runs).mean()

    def predict(self, env, board):
        eva = np.array(self.roll_out(env, board))
        print(eva)
        return eva.argmax()


if __name__ == "__main__":
    import sys

    def display(board):
        sys.stdout.write("\r")
        sys.stdout.flush()
        sys.stdout.write(board.__repr__())

    episodes = 1
    env = Environment()
    agent = MonteCarlo()
    agent2 = MonteCarlo()
    agent2.sample_size = 20
    for ii in range(episodes):
        board, score, done = env.start()
        while not done:
            action = agent.predict(env, board)
            print(board)
            print("-"*60)
            board, score, done = env.step(board, score, action)

        # for time in range(2500):
        #     action = agent.act(board)
        #     board, score, done = env.step(board, score, action)
