from dtw import dtw
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from math import sqrt
from dateutil import parser
from configparser import ConfigParser
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
import pandas as pd
import pdb
import talib
import matplotlib.pyplot as plt
import numpy as np
import ccxt
import time
import json
import ta
import collections
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from scipy.optimize import curve_fit
pd.set_option("display.precision", 8)
pd.set_option('display.max_rows', 1500)

datatype ="local"
transaction_fee = 0.00125
initial_balance = 100
BUY, SELL, HOLD = 0, 1, 2
results = {}
interval = "5m"
config = ConfigParser()
config.read("config.ini")
#exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})
profit_perc = 1.11
stoploss_perc = 0.98
image_path = "/Users/apple/Desktop/dev/projectlife/data/images/manuel/"
dateformat_save = '%Y-%m-%d-%H-%M'

def save_symbols(SYMBOLS):
	for symbol in SYMBOLS:
		data_base = exchange.fetch_ohlcv(symbol, interval,limit=960)
		df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
		symbol = symbol.split("/")[0] + symbol.split("/")[1]
		full_path = "/Users/apple/Desktop/dev/projectlife/data/allin/Binance_"+symbol+"-"+interval+".csv"
		df.to_dense().to_csv(full_path, index = False, sep=',', encoding='utf-8')
		time.sleep(1)

def backtest():
	if datatype == "all":
		exchange.load_markets()
		SYMBOLS = exchange.symbols
		save_symbols(SYMBOLS)
		for symbol in SYMBOLS:
			symbol = symbol.split("/")[0] + symbol.split("/")[1]
			df = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/allin/Binance_"+symbol+"-"+interval+".csv")
			df = df.tail(144)
			evaluate_symbol(df,symbol,SYMBOLS)
	else:
		SYMBOLS = ["NEO/BTC"]
		for symbol in SYMBOLS:
			if datatype == "remote":
				data_base = exchange.fetch_ohlcv(symbol, interval,limit=1000)
				df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
				df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
				symbol =  symbol.split("/")[0] + symbol.split("/")[1]
			elif datatype =="local":
				symbol =  symbol.split("/")[0] + symbol.split("/")[1]
				data_base = read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
				df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
			data_start_i = df[df["date"]== "2018-11-01 00:05"].index[0]
			data_end_i = df[df["date"]== "2019-07-20 00:00"].index[0]
			df = df.iloc[data_end_i-(data_end_i-data_start_i):data_end_i,:]
			df = df.reset_index()
			#df = df.tail(5000)
			evaluate_symbol(df,symbol,SYMBOLS)

def evaluate_symbol(df,symbol,SYMBOLS):
	df.set_index('date')
	df['change'] = df.close.pct_change(periods=1)*100
	df = df.fillna(0)
	df = df.drop(["volume", "index" ], axis=1)
	evaluate_trades(df,symbol,SYMBOLS)

def evaluate_trades(df,symbol,SYMBOLS):
	balance = initial_balance
	trade_count = 0
	win_count = 0
	loss_count = 0
	profit = 0
	action = HOLD
	trade_history = []
	current_tick = 0
	entry_tick = 0
	buy_mode = True
	entry_price = 0
	buy_index = 0
	for i, last_row in df.iterrows():
		current_price = last_row['close']
		current_tick += 1
		window_size = 48
		if i > window_size:
			last_row =  df.loc[i]
			prev1 =  df.loc[i-1]
			prev2 =  df.loc[i-2]
			fragment = df.iloc[i-window_size:i,:]
			fragment1 = df.iloc[i-window_size:i+1,:]
			max_close = fragment.max().close
			min_close = fragment.min().close

			# if symbol == "NEOBTC":
			#   	last_n_max_size = 0.000020
			#   	last_min_size = 0.000035
			#   	last_max_size = 0.000060

			# if symbol == "NANOBTC":
			#   	last_n_max_size = 0.0000025
			#   	last_min_size = 0.0000019
			#   	last_max_size = 0.0000100

			# if symbol == "CMTBTC":
			# 	minmax_diff_coef = 0.00000040
			# 	min_spike_diff = 0.00000010
			# 	max_spike_diff = 0.00000017

			#buy_rectg_cond = (minmax_diff <= minmax_diff_coef and xr_diff >= min_spike_diff and xr_diff <= max_spike_diff)
			buy_prev_cond1 = True
			buy_prev_cond2 = True
			buy_prev_cond3 = True
			last_candlesize_up = last_row.close - last_row.open
			last_candlesize_down = last_row.open - last_row.close
			last_n_max_size_up = last_candlesize_up / 2
			last_n_max_size_down = last_candlesize_down / 3
			last_candlesize_maxt = last_row.close# - (last_candlesize / 2)
			# last_candlesize_mint = last_row.open - (last_candlesize / 2)


			for iy, rowr in fragment.iterrows():
				if last_candlesize_up > last_n_max_size_up:
					buy_prev_cond1 = False
					break
				if last_candlesize_down > last_n_max_size_down:
					buy_prev_cond2 = False
					break
				if rowr.close > last_candlesize_maxt:
					buy_prev_cond3 = False
					break
			# 	if rowr.open < last_candlesize_mint:
			# 		buy_prev_cond3 = False
			# 		break

			#last_row_bat = (last_row.low == last_row.open)# and (last_row.high - last_row.close) > (last_candlesize/ 3) and
			#buy_cond = last_row_bat and buy_prev_cond1 and buy_prev_cond2 and buy_prev_cond3 and last_row.change < 4 and last_row.change > 1
			sell_cond = last_row.close < prev1.close

			buy_cond = buy_prev_cond1 and buy_prev_cond2 and buy_prev_cond3 and (last_row.close > max_close) and last_row.open > prev1.close and (max_close + (last_candlesize_up/2) > last_row.close) and (min_close > (last_row.open - (last_candlesize_up)))

			# if last_row.date == "2018-11-11 23:50":
			#      pdb.set_trace()

			# if last_row.date == "2019-03-31 09:25":
			#     pdb.set_trace()

			# if last_row.date == "2019-06-13 02:20":
			#     pdb.set_trace()

			# if last_row.date == "2019-07-17 16:05":
			#     pdb.set_trace()

			stopl_cond = False
			if buy_cond and buy_mode:
				print(fragment1)
				pdb.set_trace()
				buy_index = i
				action = BUY
				entry_price =  current_price
				entry_tick = current_tick
				quantity = balance / entry_price
				buy_mode = False
				print("##### TRADE " +  str(trade_count) + " #####")
				print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(last_row.date))
			elif not buy_mode and (sell_cond or stopl_cond):
				action = SELL
				if stopl_cond:
					sell_type = "stoploss"
					exit_price =  current_price
				else:
					sell_type = "profit"
					if df.iloc[buy_index+1].change > 8:
						exit_price = df.iloc[buy_index+1].close
					else:
						exit_price = current_price

				profit = ((exit_price - entry_price)/entry_price + 1)*(1-transaction_fee)**2 - 1
				balance = balance * (1.0 + profit)
				entry_price = 0
				trade_count += 1

				if profit <= 0:
					loss_count += 1
					plt.title("loss")
				else:
					plt.title("win")
					win_count += 1
				buy_mode = True
				print("buy at: "+str(buy_index))
				print("sell at: "+str(i))
				print("SELL FOR " + sell_type.upper() +" : " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(last_row.date) )
				print("PROFIT: " + str(profit*100))
				print("BALANCE: " + str(balance))
				print("==================================")
			else:
				action = HOLD

		trade_history.append((action, current_tick, current_price, balance, profit))

		if (current_tick > len(df)-1):
			results[symbol] = {'balance':np.array([balance]), "trade_history":trade_history, "trade_count":trade_count }
			print("**********************************")
			print("TOTAL BALANCE FOR "+symbol +": "+ str(balance))
			print("**********************************")
			#plot_buy_sell(trade_history)


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
	backtest()