from math import sqrt
from dateutil import parser
from configparser import ConfigParser
from pandas import Series
import matplotlib.dates as mdates
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import matplotlib
if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
	matplotlib.use("macOSX")
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np
import time
import json
import talib
import collections
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from datetime import datetime, timedelta
pd.set_option("display.precision", 9)
pd.set_option('display.max_rows', 3000)
pd.options.mode.chained_assignment = None
import platform

datatype ="local"
transaction_fee = 0.00125
initial_balance = 100
BUY, SELL, HOLD = 0, 1, 2
results = {}
interval = "1m"
config = ConfigParser()
config.read("config.ini")
profit_perc = 1.11
stoploss_perc = 0.98
dateformat_save = '%Y-%m-%d-%H-%M'
import os

def plot_symbols():
	SYMBOLS = ["FUNBTC"]
	# dir = os.listdir( "/Users/apple/Desktop/dev/projectlife/data/newticker/datacollect")
	# for s in dir:
	# 	SYMBOLS.append(s.split(".csv")[0])
	# pdb.set_trace()
	for symbol in SYMBOLS:
		data_base = read_csv("/Users/apple/Desktop/dev/projectlife/data/newticker/datacollect/"+symbol+".csv")
		df = DataFrame(data_base)
		df.columns = ['symbol','date','price_change','price_change_percent','last_price','best_bid_price','best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
		df['qav_sma100'] = df.total_traded_quote_asset_volume.rolling(100).mean()
		df['qav_sma200'] = df.total_traded_quote_asset_volume.rolling(200).mean()
		df['last_sma100'] = df.last_price.rolling(100).mean()
		df['last_sma200'] = df.last_price.rolling(200).mean()
		df['last_sma400'] = df.last_price.rolling(400).mean()
		df['last_sma600'] = df.last_price.rolling(600).mean()
		plot_whole(df)

def backtest():
	SYMBOLS = []
	dir = ""
	if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
		dir = os.listdir( "/Users/apple/Desktop/dev/projectlife/data/ticker/")
	else:
		dir = os.listdir( "/home/ubuntu/datacollect/")
	for s in dir:
		if ".py" not in s:
	 		SYMBOLS.append(s.split(".csv")[0])
	for symbol in SYMBOLS:
		trade_count = 0
		trade_history = []
		balance = initial_balance
		win_count = 0
		loss_count = 0
		profit = 0
		action = HOLD
		current_tick = 0
		entry_tick = 0
		buy_mode = True
		entry_price = 0
		buy_index = 0
		data_base = ""
		if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
			data_base = read_csv("/Users/apple/Desktop/dev/projectlife/data/ticker/"+symbol+".csv") #
		else:
			data_base = read_csv("home/ubuntu/datacollecttemp/"+symbol+".csv")
		df = DataFrame(data_base)
		df.columns = ['symbol','date','price_change','price_change_percent','last_price','best_bid_price','best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
		# df = DataFrame(data_base, columns=['symbol','date','price_change','price_change_percent','last_price','best_bid_price',
		# 		'best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume'])
		df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
		df['qav_sma100'] = df.total_traded_quote_asset_volume.rolling(100).mean()
		df['qav_sma200'] = df.total_traded_quote_asset_volume.rolling(200).mean()
		df['last_sma100'] = df.last_price.rolling(100).mean()
		df['last_sma200'] = df.last_price.rolling(200).mean()
		df['last_sma400'] = df.last_price.rolling(400).mean()
		df['last_sma600'] = df.last_price.rolling(600).mean()
		#df_x = df
		#df = df.iloc[23756:27257  ,: ]
		#df = df.iloc[343274:343775,: ]
		#fragment = detect_anomaly(df)
		#print(fragment[['symbol','last_price', 'total_traded_quote_asset_volume', 'label_qav', 'score_qav','change_qav','change_price']].tail(200))
		#plot_whole(df_x)
		#pdb.set_trace()
		df = df.reset_index()
		df = df.fillna(0)
		window_size = 500
		found_dates=[]
		found_list=[]
		for i, row in df.iterrows():
			current_price = row['last_price']
			current_tick += 1
			if i > window_size:
				fragment = df.iloc[i-window_size:i,:]
				fragment = detect_anomaly(fragment)
				fragment = fragment.reset_index()
				last =  fragment.iloc[-1,:]
				prev1 =  fragment.iloc[-2,:]
				prev2 =  fragment.iloc[-3,:]
				buy_cond1 =  (
								sum(fragment.tail(35)[-5:]['label_qav'].astype("str").str.contains("1")) == 5 and
								sum(fragment.tail(35)[:30]['label_qav'].astype("str").str.contains("0")) == 30 and
								sum(fragment.tail(35)[-5:]['change_qav']) > 0.5 and
								sum(fragment.tail(35)[-4:]['change_qav']) > 0.02 and
								sum(fragment.tail(35)[-5:]['change_price']) > 0.2 and
								fragment.iloc[-5]["score_qav"] > 0.25 and
								fragment.iloc[-5]["change_price"] > 0 and
								fragment.iloc[-5]["change_qav"] > 0 and
								(
									(fragment.iloc[-1]["change_qav"] > fragment.iloc[-5]["change_qav"]) or
									(fragment.iloc[-2]["change_qav"] > fragment.iloc[-5]["change_qav"])
								)
							)

				buy_cond2 =  (
								sum(fragment.tail(100)[:75]['label_qav'].astype("str").str.contains("0")) == 75 and
								len(fragment.tail(100)[-25:].drop_duplicates(subset="score_qav")) > 2 and
								fragment.tail(100)[-25:].drop_duplicates(subset="label_qav").label_qav.is_monotonic_increasing and
								fragment.tail(100)[-25:].drop_duplicates(subset="score_qav").score_qav.is_monotonic_increasing and
								fragment.tail(100)[-25:].drop_duplicates(subset="score_qav").change_qav.is_monotonic_increasing and
								(1 in fragment.tail(100)[-25:].drop_duplicates(subset="label_qav").label_qav.values) and
								#fragment.tail(100)[-25:].drop_duplicates(subset="score_qav").iloc[0].score_qav > 0 and
								#fragment.tail(100)[-25:].drop_duplicates(subset="score_qav").iloc[-2].change_qav > 0.5 and
								fragment.tail(100)[-25:].drop_duplicates(subset="change_qav").iloc[-1].change_qav > 1
							)

				sell_cond  =(last['last_sma600'] < prev1['last_sma600'])
				if buy_mode and (buy_cond1 or buy_cond2):
					if buy_cond1:
						print("buy cond 1 working..")
					else:
						print("buy cond 2 working..")
					buy_index = i
					action = BUY
					entry_price =  current_price
					entry_tick = current_tick
					quantity = balance / entry_price
					buy_mode = False
					print("##### TRADE " +  str(trade_count) + " #####")
					print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(last.date))
					fragment_tmp = df.iloc[i-window_size:i+20,:]
					fragment_tmp = detect_anomaly(fragment_tmp)
					print(fragment[['index','date','symbol','last_price', 'total_traded_quote_asset_volume', 'label_qav', 'score_qav','change_qav','change_price']].tail(100))
					print("20 after----->>>>>>>")
					print(fragment_tmp[['index','symbol','last_price', 'total_traded_quote_asset_volume', 'label_qav', 'score_qav','change_qav','change_price']].tail(100))
					#pdb.set_trace()
				elif not buy_mode and sell_cond:
					action = SELL
					exit_price = current_price
					profit = ((exit_price - entry_price)/entry_price + 1)*(1-transaction_fee)**2 - 1
					balance = balance * (1.0 + profit)
					entry_price = 0
					trade_count += 1
					buy_mode = True
					print("SELL: " + str(quantity) + " " +symbol+" for "+ str(exit_price) + " at " +  str(last.date))
					print("PROFIT: " + str(profit*100))
					print("BALANCE: " + str(balance))
				else:
					action = HOLD

				trade_history.append((action, current_tick, current_price, balance, profit))

				if i % 1000 == 0:
					print(symbol+"-"+str(row['index']))

				if (current_tick > len(df)-1):
					results[symbol] = {'balance':np.array([balance]), "trade_history":trade_history, "trade_count":trade_count }
					print("**********************************")
					print("TOTAL BALANCE FOR "+symbol +": "+ str(balance))
					print("TRADE COUNT FOR "+symbol +": "+ str(trade_count))
					print("**********************************")
					#plot_buy_sell(trade_history)


def plot_buy_sell(trade_history):
	closes = [data[2] for data in trade_history]
	closes_index = [data[1] for data in trade_history]
	buy_tick = np.array([data[1] for data in trade_history if data[0] == 0])
	buy_price = np.array([data[2] for data in trade_history if data[0] == 0])
	sell_tick = np.array([data[1] for data in trade_history if data[0] == 1])
	sell_price = np.array([data[2] for data in trade_history if data[0] == 1])
	plt.plot(closes_index, closes)
	plt.scatter(buy_tick, buy_price, c='g', marker="^", s=50)
	plt.scatter(sell_tick, sell_price , c='r', marker="v", s=50)
	#plt.savefig(base_path+"/results/plots/"+algo +"-"+pair+"-"+interval+"-"+str(balance)+".png" )
	plt.show(block=True)


def plot_whole(df):
	plt.clf()
	fig, axes = plt.subplots(nrows=2, ncols=1)
	df.total_traded_quote_asset_volume.plot(ax=axes[0] , color="blue", style='.-')
	df.qav_sma100.plot(ax=axes[0], color="orange")
	df.qav_sma200.plot(ax=axes[0], color="green")
	df.last_price.plot(ax=axes[1], style='.-')
	df.last_sma200.plot(ax=axes[1], color="purple")
	df.last_sma400.plot(ax=axes[1], color="red")
	df.last_sma600.plot(ax=axes[1], color="black")
	plt.title(df.iloc[-1].symbol)
	plt.show()

def plot_trades(df):
	pdb.set_trace()
	df.plot(x='close', y='mark',style='.-',linestyle='-', marker='o', markerfacecolor='black')
	trade_history.plot(x='action', y='current_price',linestyle='-', marker='o', markerfacecolor='black', plot_data_points= True)

def detect_anomaly(df):
	df = df.fillna(0)
	clf =HBOS()
	x_values = df.index.values.reshape(df.index.values.shape[0],1)
	y_values = df.total_traded_quote_asset_volume.values.reshape(df.total_traded_quote_asset_volume.values.shape[0],1)
	clf.fit(y_values)
	clf.predict(y_values)
	df["label_qav"] = clf.predict(y_values)
	df["score_qav"] = clf.decision_function(y_values)#.round(6)
	df['change_qav'] = df.total_traded_quote_asset_volume.pct_change(periods=1)*100
	df['change_price'] = df.last_price.pct_change(periods=1)*100
	return df

def plot_four_subplots():
	fig, axes = plt.subplots(nrows=2, ncols=2)
	df.total_traded_quote_asset_volume.plot(ax=axes[0,0])
	df.total_traded_base_asset_volume.plot(ax=axes[0,1])
	df.last_price.plot(ax=axes[1,0])
	df.price_change_percent.plot(ax=axes[1,1])

def pct_change(first, second):
	diff = second - first
	change = 0
	try:
		if diff > 0:
			change = (diff / first) * 100
		elif diff < 0:
			diff = first - second
			change = -((diff / first) * 100)
	except ZeroDivisionError:
		return float('inf')
	return change

def print_df(df):
	with pd.option_context('display.max_rows', None):
		print(df)


if __name__ == '__main__':
	#plot_symbols()
	backtest()

