import numpy as np
import os
import glob
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Flatten, Input, LSTM, CuDNNLSTM,Concatenate
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
import pdb
from environment import Environment
import utils


TRAIN_TIMESTEPS = 500000
PAIR = "XRPBTC"
ALGO = "DDQN"
INTERVAL ="1h"
FEATURES = ["close","sma15","sma50"]
BASE_PATH = "/Users/apple/Desktop/dev/projectlife"
FILENAME = "trajectory"+"-"+PAIR+"-"+ALGO
WINDOW_SIZE = 30

env = Environment(mode="train", interval=INTERVAL,  pair=PAIR, algo=ALGO, data_features=FEATURES)
env_validate = Environment(mode="validate", interval=INTERVAL,  pair=PAIR, algo=ALGO, data_features=FEATURES)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(LSTM(64, input_shape=env.obs_shape, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=WINDOW_SIZE)
policy = EpsGreedyQPolicy()# policy = BoltzmannQPolicy()
dqn_agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,  enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy, processor=None)
dqn_agent.compile(Adam(lr=1e-3), metrics=['mae'])

while True:
    dqn_agent.fit(env, nb_steps=9321, nb_max_episode_steps=9321, visualize=False, verbose=2)
    try:
        info = dqn_agent.test(env_validate, nb_episodes=1, visualize=False)
        trade_count, reward, balance = info['trade_count'], info['total_reward'], int(info['balance'])
        dqn_agent.save_weights(BASE_PATH+'/results/weights/{0}_{1}_wgt|bal_{2}_trc_{3}_rwd_{4}.h5'.format(ALGO, PAIR, balance, trade_count, reward),  overwrite=True)
    except KeyboardInterrupt:
        continue


