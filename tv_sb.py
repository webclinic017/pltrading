import gym
from gym import spaces
from stable_baselines import GAIL, PPO2, DQN, PPO2, DDPG
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack
from environment import Environment
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from numpy import load
import numpy as np
import pdb
import pytz
import talib
import os
import cv2
import itertools

TRAIN_TIMESTEPS = 100000
PAIR = "XRPBTC"
ALGO = "PPO2"
INTERVAL ="15m"
FEATURES = ["close","sma15"]
BASE_PATH = "/Users/apple/Desktop/dev/projectlife"
FILENAME = "trajectory"+"-"+PAIR+"-"+ALGO

def run():
	sb_lstm()

def sb_lstm():
	train_env = DummyVecEnv([lambda: Environment(mode="train", interval=INTERVAL,  pair=PAIR, algo=ALGO, data_features=FEATURES)])
	model =PPO2('MlpLstmPolicy', train_env, nminibatches=1, verbose=1)
	model.learn(TRAIN_TIMESTEPS)
	validate_env = DummyVecEnv([lambda: Environment(mode="validate", interval=INTERVAL, pair=PAIR, algo=ALGO, data_features=FEATURES)])
	obs = validate_env.reset()
	state = None
	done = [False for _ in range(validate_env.num_envs)]
	for _ in range(len(validate_env.envs[0].df)):
	    action, state = model.predict(obs, state=state, mask=done)
	    obs, reward , done, _ = validate_env.step(action)
	    validate_env.render()


if __name__ == '__main__':
	run()
