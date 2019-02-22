import random
import numpy as np
from numpy.random import choice
from collections import deque

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.optimizers import Adam

from environment.environment import legal_moves, start
from environment.movement import game_move, action_to_dir_and_ax

from os.path import join
import os

outputdir = "./output/new_setting"
if not os.path.exists(outputdir):
    os.makedirs(outputdir)


class AdvanActorCritic:
    def __init__(self, state_shape, action_size, gamma, n, outdir):
        self.input_shape = state_shape + (17,)
        self.action_size = action_size

        self.gamma = gamma**np.arange(n+1)  # dicount factor
        self.lr_policy = 0.0001
        self.lr_value = 0.0001

        self.value, self.policy = self._build_model()
        self.memory = deque(maxlen=4000)
        self.init_remember(n)
        self.n = n

        self.outdir = outdir

    def _build_model(self):
        inputs = Input(self.input_shape)
        shared = Sequential()
        shared.add(Conv2D(16, (2, 2), input_shape=self.input_shape,
                          data_format="channels_last",
                          activation="relu"))
        shared.add(Conv2D(32, (2, 2), activation="relu"))
        shared.add(Flatten())
        shared.add(Dense(20, activation="relu"))
        ishared = shared(inputs)

        vfuct = Dense(1, input_shape=[20], activation="linear")
        value = Model(input=inputs, output=vfuct(ishared))
        value.compile(loss="mse", optimizer=Adam(lr=self.lr_value))

        pfuct = Dense(self.action_size, input_shape=[20], activation="softmax")
        policy = Model(input=inputs, output=pfuct(ishared))
        policy.compile(loss="categorical_crossentropy",
                       optimizer=Adam(lr=self.lr_value))
        return value, policy

    def reshape(self, state):
        nstate = np.zeros((1, ) + state.shape + (17,))
        nstate[..., 0] = (state == 0)*1
        for ii in range(16):
            nstate[..., ii+1] = (state == 2**(ii+1))*1
        return nstate

    def act(self, state):
        legal = legal_moves(state)
        if legal.sum() == 1:
            return np.arange(4)[legal], 0
        state = self.reshape(state)
        pred = self.policy.predict(state)[0]
        a = choice(np.arange(4), p=pred)
        return a, pred

    def remember(self, state, action, reward, done):
        self.rewards.append(reward)
        self.states.append((state))
        self.action.append((action))
        self.memory.append((self.states[0], self.action[0],
                            list(self.rewards), self.states[-1]))

    def init_remember(self, n):
        self.rewards = deque(maxlen=n)
        self.states = deque(maxlen=n+1)
        self.action = deque(maxlen=n)

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for bstate, action, rewards, endstate in minibatch:
            bootstrap = self.value.predict(self.reshape(endstate))
            rewards = np.array(tuple(rewards) + (bootstrap,))
            G = (self.gamma[np.arange(len(rewards))] * rewards).sum()
            self.value.fit(self.reshape(endstate), G, epochs=1, verbose=0)
            pred = np.zeros((1, 4))
            pred[0][action] = G - self.value.predict(self.reshape(bstate))[0]
            self.policy.fit(self.reshape(bstate), pred, epochs=1, verbose=0)

    def save(self, name):
        self.value.save_weights(join(self.outdir, "value_" + name + ".hdf5"))
        self.policy.save_weights(join(self.outdir, "policy_" + name + ".hdf5"))

    def load(self, vname, pname):
        self.value.load_weights(vname)
        self.policy.load_weights(pname)


n = 30  # len till bootstrap
state, r = start()
agent = AdvanActorCritic(state.shape, 4, 0.999, n, outputdir)
# agent.load("./output/ac/value_0010.hdf5", "./output/ac/policy_0010.hdf5")

x = 20
rn = 800
k = 0

for i in range(40000):
    state, r = start()
    agent.states.append((state))
    for ii in range(rn):
        a, pred = agent.act(state)
        state, re = game_move(state, *action_to_dir_and_ax(a))
        agent.remember(state, a, re, False)
        if legal_moves(state).sum() == 0:
            break
        r += re
        print(state)
    print("sv: ", int(i/x), r, state.max())
    if state.max() >= 1000:
        if k == 0:
            agent.save("./output/first1024.hdf5")
            k += 1
            rn = 2000
        print(state)
    agent.train(20)
    agent.init_remember(agent.n)
    if i % x == 0:
        agent.save("{:04}".format(int(i/x)))
