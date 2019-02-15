import numpy as np

import random
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

from environment.environment import legal_moves, start
from environment.movement import game_move, action_to_dir_and_ax


class QValueAgent:
    def __init__(self, state_shape, action_size):
        self.input_shape = state_shape + (17,)
        self.action_size = action_size

        self.gamma = 0.9  # dicount factor
        self.learning_rate = 0.001
        self.epsilon = 0.4  # exploration rate

        self.model = self._build_model()
        self.memory = deque(maxlen=4000)

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (2, 2), input_shape=self.input_shape,
                         data_format="channels_last",
                         activation="relu"))
        model.add(Conv2D(32, (2, 2), activation="relu"))
        model.add(Flatten())
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, nextstate, done):
        self.memory.append((state, action, reward, nextstate, done))

    def act(self, state):
        legal_acts = legal_moves(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(np.arange(4)[legal_acts])
        act_values = self.model.predict(self.reshape(state))
        return np.argmax(act_values[0][legal_acts])

    def reshape(self, state):
        nstate = np.zeros((1, ) + state.shape + (17,))
        nstate[..., 0] = (state == 0)*1
        for ii in range(16):
            nstate[..., ii+1] = (state == 2**(ii+1))*1
        return nstate

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, nextstate, done in minibatch:
            qnew = reward
            if not done:
                qnew += self.gamma * np.max(self.model.predict(
                    self.reshape(nextstate)))
            state = self.reshape(state)
            q = self.model.predict(state)
            q[0][action] = qnew
            self.model.fit(state, q, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


state, score = start()
agent = QValueAgent(state.shape, 4)

for ii in range(1000000):
    a = agent.act(state)
    nstate, sc = game_move(state, *action_to_dir_and_ax(a))
    if ii % 3:
        print(state)
    if (legal_moves(nstate)).sum() == 0:
        agent.remember(state, a, sc, nstate, True)
        print(nstate)
        break
    agent.remember(state, a, sc, nstate, False)
    score += sc
    state = nstate
agent.train(19)
