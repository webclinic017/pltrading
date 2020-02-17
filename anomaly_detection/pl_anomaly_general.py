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
from sklearn import preprocessing
from datetime import datetime, timedelta
#pd.set_option("display.precision", 8)
#pd.set_option('display.max_rows', 5000)

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

def save_symbols(SYMBOLS):
	for symbol in SYMBOLS:
		data_base = exchange.fetch_ohlcv(symbol, interval,limit=960)
		df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
		symbol = symbol.split("/")[0] + symbol.split("/")[1]
		full_path = "/Users/apple/Desktop/dev/projectlife/data/allin/Binance_"+symbol+"-"+interval+".csv"
		df.to_dense().to_csv(full_path, index = False, sep=',', encoding='utf-8')
		time.sleep(1)

def backtest(start_profit):
	SYMBOLS = ["ARN/BTC", "DLT/BTC","BCD/BTC","OMG/BTC","ONE/BTC","ONG/BTC","ONT/BTC","OST/BTC","PHB/BTC","PIVX/BTC","POA/BTC","HOT/BTC","ICX/BTC","KEY/BTC","KNC/BTC","TRX/BTC","STORJ/BTC","MTH/BTC","RLC/BTC","RCN/BTC","NANO/BTC", "NEO/BTC", "GAS/BTC", "OAX/BTC", "TNT/BTC", "POE/BTC"]
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

		### second check start ####
		selected_dates = []
		if symbol == "ARNBTC":
			selected_dates = ["2018-02-12 15:59",
						"2018-04-08 13:30",
						"2018-07-09 14:13",
						"2018-08-17 18:05",
						"2018-09-26 19:19",
						"2018-11-29 11:34",
						"2019-01-17 01:14",
						"2019-04-03 09:12",
						"2019-04-22 05:14",
						"2019-05-04 14:43",
						"2019-06-21 19:00",
						"2019-07-02 09:42",
						"2019-07-25 01:32",
						"2019-08-01 07:38",
						"2020-01-19 20:39"]
		elif symbol == "ARKBTC": #******
			selected_dates = ["2018-02-12 15:59",
						"2019-09-03 22:59",
						"2019-02-10 18:31"]
		elif symbol == "HBARBTC": #******
			selected_dates = ["2019-09-30 02:13" ]
		elif  symbol == "BCDBTC":
			selected_dates = ["2018-01-03 06:46",
						"2018-01-09 05:00",
						"2018-01-09 06:30",
						"2018-01-10 08:50",
						"2018-01-11 07:18",
						"2018-05-06 04:34",
						"2018-07-09 15:39",
						"2018-08-30 13:08",
						"2018-09-05 06:33",
						"2018-10-23 12:33"]
		elif  symbol == "DLTBTC":
			selected_dates = ["2018-01-06 17:14",
						"2018-01-08 19:00",
						"2018-01-10 07:40",
						"2018-01-16 06:31",
						"2018-01-22 19:32",
						"2018-01-22 23:14",
						"2018-01-31 19:59",
						"2018-02-10 16:00",
						"2018-02-25 22:31",
						"2018-03-11 20:00",
						"2018-05-08 15:59",
						"2018-05-10 15:30",
						"2018-05-28 22:50",
						"2018-06-10 07:00",
						"2018-06-16 23:57",
						"2018-07-03 16:00",
						"2018-07-10 13:33",
						"2018-07-19 14:00",
						"2018-07-23 12:40",
						"2018-08-07 09:01",
						"2018-08-15 12:00",
						"2018-09-14 19:27",
						"2018-09-16 00:03",
						"2018-10-30 20:18",
						"2019-04-11 15:17",
						"2019-05-05 16:00"]
		elif  symbol == "GASBTC":
			selected_dates = ["2018-01-03 01:29",
						"2018-01-05 23:27",
						"2018-01-19 22:08",
						"2018-05-31 18:31",
						"2018-06-24 02:19",
						"2018-07-03 23:38",
						"2018-08-16 04:07",
						"2018-08-20 06:37",
						"2018-08-23 15:33",
						"2018-09-09 11:57",
						"2018-09-09 22:54",
						"2018-09-28 03:32",
						"2018-11-20 21:52",
						"2018-12-02 00:02",
						"2018-12-07 05:35",
						"2019-01-17 00:46",
						"2019-02-17 00:12",
						"2019-02-20 00:11",
						"2019-03-13 00:15",
						"2019-03-22 00:00",
						"2019-05-27 09:53",
						"2019-06-18 11:15",
						"2019-06-19 00:00",
						"2019-07-10 08:42",
						"2019-07-17 11:29",
						"2019-07-24 03:32",
						"2019-08-07 00:16"]
		elif  symbol == "ICXBTC":
			selected_dates = ["2018-06-13 07:40"]
		elif  symbol == "KEYBTC":
			selected_dates = ["2019-07-04 21:20"]
		elif  symbol == "KNCBTC":
			selected_dates = ["2018-01-01 00:56",
						"2018-01-03 16:51",
						"2018-01-06 11:23",
						"2018-01-17 10:13",
						"2018-01-26 16:31",
						"2018-03-24 18:04",
						"2018-04-09 04:00",
						"2018-06-28 15:35",
						"2018-08-21 16:59",
						"2018-10-26 03:30",
						"2019-02-08 00:27",
						"2019-02-10 09:48",
						"2019-03-25 20:44",
						"2019-04-11 15:02",
						"2019-05-12 16:52",
						"2019-05-14 09:01",
						"2019-05-23 00:18",
						"2019-05-28 15:19",
						"2019-06-24 18:29",
						"2019-06-25 17:00",
						"2019-06-26 09:51",
						"2019-08-07 10:24"]
		elif  symbol == "MTHBTC":
			selected_dates = ["2018-01-16 01:53",
						"2018-02-05 15:24",
						"2018-03-07 09:44",
						"2018-03-09 13:42",
						"2018-03-14 14:05",
						"2018-03-16 05:41",
						"2018-03-24 09:47",
						"2018-04-19 08:00",
						"2018-06-18 06:28",
						"2018-07-01 06:57",
						"2018-07-20 15:59",
						"2018-07-22 06:00",
						"2018-08-07 08:05",
						"2018-08-11 06:46",
						"2019-05-06 02:39",
						"2019-05-16 16:29",
						"2019-05-20 17:29",
						"2019-05-22 18:25",
						"2019-08-16 18:46",
						"2019-08-17 00:51",
						"2019-08-27 19:53",
						"2019-08-29 03:58"]
		elif  symbol == "NANOBTC":
			selected_dates = ["2018-03-22 19:07",
						"2018-05-03 08:24",
						"2018-06-14 09:29",
						"2019-07-23 21:20",
						"2019-07-29 19:06"]
		elif  symbol == "NEOBTC":
			selected_dates = ["2018-07-04 08:00",
						"2019-05-15 13:00",
						"2019-08-03 23:41",
						"2019-08-11 22:38"]
		elif  symbol == "OAXBTC":
			selected_dates = ["2018-01-12 05:48",
						"2018-01-12 18:00",
						"2018-01-15 11:09",
						"2018-01-18 16:12",
						"2018-01-24 20:00",
						"2018-02-15 20:55",
						"2018-02-20 05:22",
						"2018-02-20 07:26",
						"2018-02-23 01:00",
						"2018-02-27 10:16",
						"2018-03-29 07:00",
						"2018-04-05 05:28",
						"2018-04-29 19:00",
						"2018-05-17 15:00",
						"2018-07-09 15:00",
						"2018-07-28 18:55",
						"2018-08-24 16:58",
						"2018-08-30 19:32",
						"2018-09-03 10:26",
						"2018-09-25 14:00",
						"2018-09-25 20:17",
						"2018-10-09 18:59",
						"2018-11-17 22:00",
						"2018-12-07 17:00",
						"2019-01-20 18:06",
						"2019-06-14 18:24"]
		elif  symbol == "OMGBTC":
			selected_dates = ["2018-03-23 23:05",
						"2018-04-13 01:48",
						"2018-05-30 03:47",
						"2018-11-19 15:31",
						"2018-12-06 11:57",
						"2018-12-13 16:59",
						"2019-04-02 08:10",
						"2019-05-08 15:28"]
		elif  symbol == "ONGBTC":
			selected_dates = ["2019-03-12 19:56",
						"2019-03-13 08:19",
						"2019-03-27 10:54",
						"2019-05-01 05:02",
						"2019-05-15 21:13",
						"2019-05-20 16:00",
						"2019-05-23 14:36",
						"2019-05-24 03:29",
						"2019-06-06 00:00",
						"2019-07-12 17:25",
						"2019-07-26 01:25",
						"2019-08-03 02:56",
						"2019-08-19 15:59"]
		elif  symbol == "ONGBTC":
			selected_dates = ["2018-05-03 01:04",
				"2019-07-11 10:36",
				"2019-08-28 04:53"]
		elif  symbol == "OSTBTC":
			selected_dates = ["2018-01-01 03:46",
				"2018-01-02 03:47",
				"2018-01-14 10:15",
				"2018-03-18 15:29",
				"2018-06-17 02:57",
				"2018-06-22 10:07",
				"2018-07-15 21:01",
				"2018-07-22 02:08",
				"2018-08-07 07:16",
				"2018-08-22 01:06",
				"2018-08-22 13:59",
				"2018-10-12 00:08",
				"2018-11-06 18:46",
				"2019-02-10 11:18",
				"2019-02-17 02:26",
				"2019-03-11 14:36",
				"2019-03-27 00:01",
				"2019-03-28 00:00",
				"2019-06-03 13:20",
				"2019-06-20 00:00"]
		elif  symbol == "PHBBTC":
			selected_dates = ["2019-06-24 20:52",
				"2019-07-04 19:33"]
		elif  symbol == "PIVXBTC":
			selected_dates = ["2018-02-01 15:54",
				"2018-02-15 08:23",
				"2018-02-18 23:19",
				"2018-03-15 07:06",
				"2018-03-27 08:08",
				"2018-06-19 16:52",
				"2018-06-27 07:36",
				"2018-08-13 16:34",
				"2018-08-22 00:19",
				"2018-10-18 01:01",
				"2018-11-21 20:35",
				"2019-01-10 20:52",
				"2019-02-08 13:29",
				"2019-02-13 00:26",
				"2019-05-17 00:00",
				"2019-06-02 16:00",
				"2019-07-01 00:10",
				"2019-07-02 15:59",
				"2019-07-05 15:59",
				"2019-07-15 09:34",
				"2019-08-28 02:54",
				"2019-08-31 00:00"]
		elif  symbol == "POABTC":
			selected_dates = ["2018-03-09 10:28",
							"2018-06-13 07:08",
							"2018-07-01 10:49",
							"2018-07-03 21:16",
							"2018-07-29 02:58",
							"2018-08-02 05:57",
							"2018-08-13 20:13",
							"2018-09-03 10:55",
							"2018-12-01 13:09",
							"2018-12-29 17:00",
							"2019-05-05 19:00",
							"2019-05-10 12:00",
							"2019-05-25 17:00",
							"2019-05-26 21:00",
							"2019-06-01 04:18",
							"2019-08-06 15:59",
							"2019-08-12 07:17",
							"2019-08-12 11:30",
							"2019-08-26 23:00"]
		elif  symbol == "RCNBTC":
			selected_dates = ["2018-01-06 07:09",
							"2018-01-28 19:35",
							"2018-03-06 00:01",
							"2018-03-09 01:21",
							"2018-09-20 19:23",
							"2019-01-01 16:37",
							"2019-02-05 11:34",
							"2019-02-28 22:04",
							"2019-06-20 20:11",
							"2019-07-04 11:21",
							"2019-07-10 16:00",
							"2019-07-16 22:13",
							"2019-07-22 16:00",
							"2019-07-26 12:58",
							"2019-08-10 16:00",
							"2019-08-18 16:00",
							"2019-08-20 12:25",
							"2019-08-25 13:29"]
		elif  symbol == "RLCBTC":
			selected_dates = ["2018-01-21 17:59",
							"2018-02-02 08:18",
							"2018-02-25 21:43",
							"2018-02-26 07:27",
							"2018-04-25 00:00",
							"2018-08-15 03:52",
							"2018-08-28 16:23",
							"2019-02-28 17:34",
							"2019-03-28 13:06",
							"2019-03-31 12:17",
							"2019-04-26 14:08",
							"2019-05-03 21:24",
							"2019-06-15 15:09",
							"2019-08-06 18:58",
							"2019-08-31 06:45"]
		elif  symbol == "STORJBTC":
			selected_dates = ["2018-01-24 17:29",
							"2018-02-02 16:31",
							"2018-02-15 07:39",
							"2018-02-16 00:01",
							"2018-02-21 00:00",
							"2018-11-19 00:50",
							"2018-12-08 18:45",
							"2018-12-14 16:24",
							"2018-12-17 00:00",
							"2018-12-21 00:02",
							"2019-02-04 16:29",
							"2019-02-27 00:00",
							"2019-04-30 00:00",
							"2019-06-11 18:00",
							"2019-07-25 21:03",
							"2019-08-21 18:00",
							"2019-08-30 14:15"]
		elif  symbol == "TNTBTC":
			selected_dates = ["2018-03-20 16:38",
							"2018-03-25 18:33",
							"2018-04-14 17:00",
							"2018-06-23 15:59",
							"2018-07-11 19:48",
							"2018-11-30 07:12",
							"2018-12-28 05:58",
							"2019-01-06 20:59"]
		else:
			continue

		for date in selected_dates:
			df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
			time = datetime.strptime(date,'%Y-%m-%d %H:%M')
			start = format((time + timedelta(hours=-1)),'%Y-%m-%d %H:%M')
			end = format((time + timedelta(hours=+1)),'%Y-%m-%d %H:%M')
			try:
				data_start_i = df[df["date"]== start].index[0]
				data_end_i = df[df["date"]== end].index[0]
				df = df.iloc[data_end_i-(data_end_i-data_start_i):data_end_i,:]
				df = df.reset_index()
				df.set_index('date')
				### second check end ####

				df['change'] = df.close.pct_change(periods=1)*100
				df = df.fillna(0)
				df = detect_anomaly(df)
				df = df.drop(["volume"], axis=1) #"high","low",
				evaluate_trades(df,symbol,SYMBOLS, start_profit)
			except:
				#pdb.set_trace()
				print("pass " + symbol + date)



def normalize_data(df):
	x = df.close.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
	df.close = pd.DataFrame(x_scaled)
	return df


def plot_fragment(fragment,symbol,date, folder):
	#pdb.set_trace()
	plt.clf()
	fragment = fragment.reset_index()
	plt.figure()
	plt.plot(fragment.close,label="close", marker="o")
	plt.axis('off')
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	frame1 = plt.gca()
	frame1.axes.xaxis.set_ticklabels([])
	frame1.axes.yaxis.set_ticklabels([])
	main_path = "/Users/apple/Desktop/dev/projectlife/data/images/"
	path = main_path + folder + "/"+symbol+"-"+str(date)+"-"+".png"
	plt.savefig(path)

def detect_anomaly(df):
	x_values = df.index.values.reshape(df.index.values.shape[0],1)
	y_values = df.change.values.reshape(df.change.values.shape[0],1)
	clf = KNN()
	clf.fit(y_values)
	clf.predict(y_values)
	df["label_knn"] = clf.predict(y_values)
	df["score_knn"] = clf.decision_function(y_values).round(4)
	return df

def evaluate_trades(df,symbol,SYMBOLS, start_profit):
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
		window_size = 44
		part_size = 10
		stoploss_perc = 5
		if i > window_size:
			last_row =  df.loc[i]
			prev1 =  df.loc[i-1]
			fragment = df.iloc[i-window_size:i,:]
			fragment1 = df.iloc[i-window_size:i+20,:]
			fragment1 = fragment1.drop(["high","low"], axis=1)
			fragment2 = df.iloc[i-window_size:i+1,:]
			last_candlesize = (last_row.close - last_row.open)
			#inverse_hammer = (last_row.close > last_row.open and (last_row.high - last_row.close) > (last_candlesize / 10)) # last_row.low == last_row.open
			# buy_past = True
			# for indexr,rowr in fragment.tail(6).iterrows():
			# 	if  rowr['label_knn'] == 1: # or rowr['label_cblof'] == 1 or rowr['label_iforest'] == 1:
			# 		buy_past = False
			# 		break

			buy_prev_candlesize = True
			for iy, rowr in fragment.iterrows():
				candle_size = abs(rowr.close - rowr.open)
				part_lastcandle = last_candlesize / part_size #### normalde 3 olacak!!!!! 2. kontrol için 5 yaptım.
				max_last = last_row.open + part_lastcandle
				min_last = last_row.open - part_lastcandle
				if candle_size > part_lastcandle:
					buy_prev_candlesize = False
					break
				if (rowr.close > max_last) or (rowr.open > max_last) or (rowr.close < min_last) or (rowr.open < min_last):
					buy_prev_candlesize = False
					break

			buy_cond = (buy_prev_candlesize and last_row['label_knn'] == 1 and last_row.change > 1 and last_row.score_knn > 0.01)
			#sell_cond = (last_row['label_knn'] == 0 ) # -150 kaybettiriyor.
			sell_cond = (last_row['change'] < 0 ) #1* partta - 90, 8 part'da -108 kaybettiriyor 6 da - 173. HEP KAYBETTİRİYR.

			# sell_for_stop_loss = False
			# if not buy_mode:
			# 	#pdb.set_trace()
			# 	if current_price > entry_price:
			# 		stoploss_price = current_price - (current_price * stoploss_perc) / 100
			# 	if current_price < stoploss_price:
			# 		sell_for_stop_loss = True

			# sell_cond =  sell_for_stop_loss

			# if (last_row.date == "2018-09-26 19:15"):
			#  	print(fragment1)

			if buy_mode and buy_cond:
				#print(fragment1)
				#plot_fragment(fragment2,symbol,last_row.date, "anomalies_1m_"+str(part_size))
				buy_index = i
				action = BUY
				entry_price =  current_price
				entry_tick = current_tick
				quantity = balance / entry_price
				buy_mode = False
				stoploss_price =  entry_price - (entry_price * stoploss_perc) / 100
				#print("##### TRADE " +  str(trade_count) + " #####")
				#print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(last_row.date))
			elif not buy_mode and sell_cond:
				action = SELL
				sell_type = "profit"
				stoploss_price = 0
				exit_price = current_price
				profit = ((exit_price - entry_price)/entry_price + 1)*(1-transaction_fee)**2 - 1
				start_profit = start_profit + profit
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
				#print("buy at: "+str(buy_index))
				#print("sell at: "+str(i))
				#print("SELL FOR " + sell_type.upper() +" : " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(last_row.date) )
				print("PROFIT: " + str(profit*100))
				#print("BALANCE: " + str(balance))
				#print("==================================")
			else:
				action = HOLD

		trade_history.append((action, current_tick, current_price, balance, profit))

		# if (current_tick > len(df)-1):
		# 	pdb.set_trace()
		# 	results[symbol] = {'balance':np.array([balance]), "trade_history":trade_history, "trade_count":trade_count }
		# 	print("**********************************")
		# 	print("TOTAL BALANCE FOR "+symbol +": "+ str(balance))
		# 	print("**********************************")

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


def angle_between(p1, p2):
	ang1 = np.arctan2(*p1[::-1])
	ang2 = np.arctan2(*p2[::-1])
	return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def calculate_angle():
	#https://www.geogebra.org/m/jmMXpzuM
	A = (0, 0) # (1, 0)
	B = (15, 6)#(1, -1) 45derece
	print(angle_between(B, A))


if __name__ == '__main__':
	start_profit = 0
	backtest(start_profit)
	print("FINAL PROFIT: " +str(start_profit))
	#calculate_angle()


##### NOTLAR #######
#1m için 55 taneden 44 tanesi artı 11 tanesi eksi çıkarıyor.
#5m için 67 taneden 49 tanesi artı 18 tanesi eksi çıkarıyor.
