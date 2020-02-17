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
import numpy as np
import ccxt
import time
import json
import collections
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from datetime import datetime, timedelta
pd.set_option("display.precision", 8)
pd.set_option('display.max_rows', 3000)
pd.options.mode.chained_assignment = None

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


def backtest(start_profit):
	SYMBOLS = ["LOOM/BTC"] #ICX
	for symbol in SYMBOLS:
		symbol =  symbol.split("/")[0] + symbol.split("/")[1]
		data_base = read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
		df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		df['change'] = df.close.pct_change(periods=1)*100
		df = df.fillna(0)

		#df = df.iloc[9500:10187, : ]
		#df_new = detect_anomaly(df, "hbos").tail(200)
		#pdb.set_trace()
		#df = df.reset_index() # 2020-01-31 09:31

		window_size = 500 # 500 #
		found_dates=[]
		found_list=[]
		for i, row in df.iterrows():
			if i > window_size:
				fragment = df.iloc[i-window_size:i,:]
				fragment = detect_anomaly(fragment, "hbos")
				fragment = fragment.reset_index()
				last =  fragment.iloc[-1,:]
				prev1 =  fragment.iloc[-2,:]
				# buy_cond = (last['label_volume'] == 1 and prev1['label_volume'] == 1 and
				# 	        last['label_close'] == 1 and   prev1['label_close'] == 1 and
				# 	        last['score_volume'] > prev1['score_volume'] and
				# 	        last['score_volume'] < prev1['score_volume'] * 20  and
				# 	        last['change'] > 0 and prev1['change'] > 0)

				# buy_cond = (last['label_volume'] == 1  and
				# 	        last['volume'] > prev1['volume'] * 5 and
				# 	        last['volume'] < prev1['volume'] * 30  and
				# 	        prev['change'] != 0 and
				# 	        last['change'] > 0 and
				# 	        sum(fragment.tail(50)['label_volume'].astype("str").str.contains("1")) == 1 and
				# 	        sum(fragment.tail(5)['change']) > 0.5)

				# buy_cond = (last['label_volume'] == 1  and ## yapmaya çalıştıpın buna cevap vermek: https://prnt.sc/qyxfk0
				# 	        last['volume'] > prev1['volume'] * 3 and
				# 	        last['volume'] < prev1['volume'] * 30  and
				# 	        last['change'] > 0 and
				# 	        prev1['change'] < 1 and  last['change'] < 1 and
				# 	        last['change'] > prev1['change'] and
				# 	        sum(fragment.tail(50).head(48)['label_volume'].astype("str").str.contains("1")) == 0  )

				# buy_cond = (last['label_volume'] == 1  and ## yapmaya çalıştıpın buna cevap vermek: https://prnt.sc/qyxfk0
				#  	        last['change'] > 0 and
				#  	        sum(fragment.tail(20)['label_close'].astype("str").str.contains("0")) == 0 and
				#  	        sum(fragment.tail(20).head(18)['label_volume'].astype("str").str.contains("1")) == 0  )

				buy_cond = (last['label_volume'] == 1  and ## yapmaya çalıştıpın buna cevap vermek: https://prnt.sc/qyxfk0
				 	        last['label_close'] == 1 and
				 	        last['change'] > 0 and
				 	        last['volume'] > prev1['volume'] * 3 and
				 	        sum(fragment.tail(10)['label_close'].astype("str").str.contains("0")) == 9 and
				 	        sum(fragment.tail(10)['label_volume'].astype("str").str.contains("0")) == 9 )


				print(last['date'])
				#if last['date'] == "2019-12-12 01:45":
				#	pdb.set_trace()
				if buy_cond:
					if last['date'] not in found_dates:
						found_dates.append(last['date'])
						found_list.append({last['date']: fragment.tail(200)})
		print(found_dates)
		print(found_list)




def plot_fragment(fragment,symbol,date, folder):
	plt.clf()
	fragment = fragment.reset_index()
	plt.figure()
	plt.plot(fragment.close,label="close", marker="o")
	plt.axis('off')
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	frame1 = plt.gca()
	frame1.axes.xaxis.set_ticklabels([])
	frame1.axes.yaxis.set_ticklabels([])

def detect_anomaly(df, type):
	clf =HBOS() #
	if type == "forest":
		clf =IForest()


	x_values = df.index.values.reshape(df.index.values.shape[0],1)
	y_values = df.close.values.reshape(df.close.values.shape[0],1)
	clf.fit(y_values)
	clf.predict(y_values)
	df["label_close"] = clf.predict(y_values)
	df["score_close"] = clf.decision_function(y_values)#.round(6)

	y_values = df.volume.values.reshape(df.volume.values.shape[0],1)
	clf.fit(y_values)
	clf.predict(y_values)
	df["label_volume"] = clf.predict(y_values)
	df["score_volume"] = clf.decision_function(y_values)#.round(4)

	# x_values = df.index.values.reshape(df.index.values.shape[0],1)
	# y_values = df.close.values.reshape(df.close.values.shape[0],1)
	# clf = KNN()
	# clf.fit(y_values)
	# clf.predict(y_values)
	# df["label_close_knn"] = clf.predict(y_values)
	# df["score_close_knn"] = clf.decision_function(y_values)#.round(6)

	# y_values = df.volume.values.reshape(df.volume.values.shape[0],1)
	# clf = KNN()
	# clf.fit(y_values)
	# clf.predict(y_values)
	# df["label_volume_knn"] = clf.predict(y_values)
	# df["score_volume_knn"] = clf.decision_function(y_values)#.round(4)

	# x_values = df.index.values.reshape(df.index.values.shape[0],1)
	# y_values = df.close.values.reshape(df.close.values.shape[0],1)
	# clf = PCA()
	# clf.fit(y_values)
	# clf.predict(y_values)
	# df["label_close_pca"] = clf.predict(y_values)
	# df["score_close_pca"] = clf.decision_function(y_values)#.round(6)

	# y_values = df.volume.values.reshape(df.volume.values.shape[0],1)
	# clf = PCA()
	# clf.fit(y_values)
	# clf.predict(y_values)
	# df["label_volume_pca"] = clf.predict(y_values)
	# df["score_volume_pca"] = clf.decision_function(y_values)#.round(4)


	# x_values = df.index.values.reshape(df.index.values.shape[0],1)
	# y_values = df.close.values.reshape(df.close.values.shape[0],1)
	# clf = IForest()
	# clf.fit(y_values)
	# clf.predict(y_values)
	# df["label_close_iforest"] = clf.predict(y_values)
	# df["score_close_iforest"] = clf.decision_function(y_values)#.round(6)

	# y_values = df.volume.values.reshape(df.volume.values.shape[0],1)
	# clf = IForest()
	# clf.fit(y_values)
	# clf.predict(y_values)
	# df["label_volume_iforest"] = clf.predict(y_values)
	# df["score_volume_iforest"] = clf.decision_function(y_values)#.round(4)

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
			buy_cond = (last_row['label_knn_volume'] == 1 and prev1['label_knn_volume'] == 1)#  last_row.change > 1 and last_row.score_knn > 0.01)
			sell_cond = (last_row['change'] < 0 )
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


if __name__ == '__main__':
	start_profit = 0
	backtest(start_profit)
	print("FINAL PROFIT: " +str(start_profit))

