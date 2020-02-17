import gym
import gym.spaces as spaces
from gym.utils import seeding
import pandas as pd
import numpy as np
import ccxt
import pdb
import sys
import talib
import matplotlib.pyplot as plt

base_path = "/Users/apple/Desktop/dev/projectlife"
continuous_envs=["DDPG",'SAC','A2C','HER']
transaction_fee = 0.00125
initial_balance = 100
LONG, FLAT = 0, 1
BUY, SELL, HOLD = 0, 1, 2

class Environment(gym.Env):

	def __init__(self, pair, algo, mode, interval, data_features, input_dim="3d", time_step=30):

		self.mode = mode
		self.pair = pair
		self.algo = algo
		#self.max_episode_steps = 1000000
		self.data_features = data_features
		self.interval = interval
		self.time_step = time_step
		self.input_dim = input_dim
		self.load_data()
		self.positions = [LONG, FLAT]
		self.actions = [BUY, SELL, HOLD]
		self.n_features = self.df.shape[1]
		self.n_actions = len(self.actions)

		if self.input_dim == "3d":
			self.obs_shape = (self.time_step, self.n_features)
		else:
			self.obs_shape = (self.n_features,)
		self.observation_space = spaces.Box(low=-self.n_features, high=self.n_features, shape=self.obs_shape, dtype=np.float32)

		self.act_shape = (self.n_actions,)
		if self.algo in continuous_envs:
			self.action_space = spaces.Box(low=-self.n_actions, high=self.n_actions, shape=self.act_shape,  dtype=np.float32)
			#self.action_space = spaces.Box(low=np.array([-len(self.actions), -1]), high=np.array([len(self.actions), 1]),  dtype=np.float32)
		else:
			self.action_space = spaces.Discrete(self.n_actions)

	def load_data(self):
		if self.mode == "train":
			raw_df = pd.read_csv(base_path+"/data/train/Binance_"+self.pair+"-"+self.interval+".csv")
		elif self.mode == "validate":
			raw_df = pd.read_csv(base_path +"/data/validate/Binance_"+self.pair+"-"+self.interval+".csv")
		elif self.mode == "trade":
			exchange = ccxt.binance({'apiKey': "x", 'secret': "y"})
			data = exchange.fetch_ohlcv(self.pair, self.interval)
			raw_df = DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

		extractor = Features(raw_df)
		self.df = extractor.add_ta_features()
		self.df.dropna(inplace=True)
		self.current_prices = self.df['close'].values
		self.df = self.df[self.data_features].values

	def reset(self):
		self.current_tick = 0
		self.trade_count = 0
		self.history = []
		self.balance = initial_balance
		self.profit = 0
		self.action = HOLD
		self.position = FLAT
		self.done = False
		self.updateState()

		return self.state

	def step(self, action):

		if self.done:
			return self.state, self.reward, self.done, {}

		self.reward = 0
		profit = 0
		self.action = HOLD
		if self.algo in continuous_envs:
		 	action = np.argmax(action)

		if action == BUY:
			if self.position == FLAT:
				self.position = LONG
				self.action = BUY
				self.entry_price = self.current_price
				self.quantity = self.balance / self.current_price

		elif action == SELL:
			if self.position == LONG:
				self.position = FLAT
				self.action = SELL
				self.exit_price = self.current_price
				profit = ((self.exit_price - self.entry_price)/self.entry_price + 1)*(1-transaction_fee)**2 - 1
				self.reward += profit
				self.balance = self.balance * (1.0 + profit)
				self.entry_price = 0
				self.trade_count += 1

		self.current_tick += 1
		self.history.append((self.action, self.current_tick, self.current_price, self.balance, self.reward))
		self.updateState()
		self.results = {'balance':np.array([self.balance]), "history":self.history, "trade_count":self.trade_count }

		if (self.current_tick > (self.df.shape[0]) - self.time_step-1):
			self.done = True
			self.reward = profit
		#self.render()

		return self.state, self.reward, self.done, self.results

	def updateState(self):
		def one_hot_encode(x, n_classes):
			return np.eye(n_classes)[x]
		self.current_price = float(self.current_prices[self.current_tick])
		prev_position = self.position
		one_hot_position = one_hot_encode(prev_position,3)
		profit = self.get_profit()
		#self.state = np.concatenate((self.df[self.current_tick], one_hot_position, [profit]))
		self.state = self.df[self.current_tick]
		return self.state

	def get_profit(self):
		if(self.position == LONG):
			profit = ((self.current_price - self.entry_price)/self.entry_price + 1)*(1-transaction_fee)**2 - 1
		else:
			profit = 0
		return profit

	def render(self, mode='human', verbose=False):
		if self.action == BUY:
			print("TICK: {0} / TRADE: {1}".format(self.current_tick, self.trade_count))
			print("--------------------------------")
			print("BUY: " + str(self.quantity) + " " +self.pair+" for "+ str(self.entry_price) + " each." )
		elif self.action == SELL:
			print("SELL: " + str(self.quantity) + " " +self.pair+" for "+ str(self.exit_price) + " each." )
			print("REWARD: {0}".format(self.reward))
			print("BALANCE: " + str(self.balance))
			print("==================================")

		if self.done == True:
			print("**********************************")
			print("TOTAL BALANCE: " + str(self.balance))
			self.plot_episode()

	def plot_episode(self):
	    closes = [data[2] for data in self.history]
	    closes_index = [data[1] for data in self.history]
	    buy_tick = np.array([data[1] for data in self.history if data[0] == 0])
	    buy_price = np.array([data[2] for data in self.history if data[0] == 0])
	    sell_tick = np.array([data[1] for data in self.history if data[0] == 1])
	    sell_price = np.array([data[2] for data in self.history if data[0] == 1])
	    plt.plot(closes_index, closes)
	    plt.scatter(buy_tick, buy_price, c='g', marker="^", s=20)
	    plt.scatter(sell_tick, sell_price , c='r', marker="v", s=20)
	    #plt.savefig(base_path+"/results/plots/"+self.algo +"-"+self.pair+"-"+self.interval+"-"+str(self.balance)+".png" )
	    #plt.show(block=True)


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

class Features:
    def __init__(self, df):
        self.df = df
        self.open = df['open'].astype('float')
        self.close = df['close'].astype('float')
        self.high = df['high'].astype('float')
        self.low = df['low'].astype('float')
        self.volume =  df['volume'].astype('float')

    def add_ta_features(self):
        self.df['sma5'] = talib.SMA(self.close,5)
        self.df['sma10'] = talib.SMA(self.close,10)
        self.df['sma15'] = talib.SMA(self.close,15)
        self.df['sma20'] = talib.SMA(self.close,20)
        self.df['sma50'] = talib.SMA(self.close,50)
        # self.df['sma200'] = talib.SMA(self.close,200)
        # self.df['rsi'] = talib.RSI(self.close, 14)
        # self.df['adx'] = talib.ADX(self.high, self.low, self.close, timeperiod=14)
        # self.df['atr'] = talib.ATR(self.high, self.low, self.close, timeperiod=14)
        return self.df
