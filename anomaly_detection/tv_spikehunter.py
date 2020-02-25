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
		SYMBOLS = ["CMT/BTC"]
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
			data_start_i = df[df["date"]== "2019-01-01 00:00"].index[0]
			data_end_i = df[df["date"]== "2019-08-05 00:00"].index[0]
			df = df.iloc[data_end_i-(data_end_i-data_start_i):data_end_i,:]
			df = df.reset_index()
			#df = df.tail(5000)
			evaluate_symbol(df,symbol,SYMBOLS)

def evaluate_symbol(df,symbol,SYMBOLS):
	df.set_index('date')
	df['change'] = df.close.pct_change(periods=1)*100
	#df["rmean24"] = df.change.rolling(24).mean()
	#df["rmean72"] = df.close.rolling(72).mean()
	#df['inv_hammer'] = talib.CDLINVERTEDHAMMER(df.open.values,df.high.values,df.low.values,df.close.values)
	#df["rmean72"] = df.change.rolling(72).mean()
	#df["cmean24"] = df.close.rolling(24).mean()
	#df["cmean48"] = df.close.rolling(48).mean()
	#df["cmean72"] = df.close.rolling(72).mean()
	#df['vpt'] = ta.volume.volume_price_trend(df["close"], df["volume"], fillna=False).round(4)
	#df["vpt_change"] =  (df.vpt.pct_change(periods=1)*100).round(4)
	#df['ema5'] = talib.EMA(df['close'],5)
	#df['ema8'] = talib.EMA(df['close'],8)
	#df['ema13'] = talib.EMA(df['close'],13)
	#df["max72"] = df.vpt.rolling(72).max().round(4)
	#df["min72"] = df.vpt.rolling(72).min().round(4)
	#pdb.set_trace()
	#df["maxmin_diff"] = df["max72"] - df["min72"]
	#df["maxmin_mean"] = (df["min72"] + df["min72"]) / 2
	df = df.fillna(0)
	df = df.drop(["volume", "index" ], axis=1)
	df = detect_anomaly(df)
	#df = detect_dtw(df)
	#plot_fragment(df,symbol)
	evaluate_trades(df,symbol,SYMBOLS)
	#fragment = df.iloc[i-window_size:i+8,:]

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
		window_size = 64
		if i > window_size:
			last_row =  df.loc[i]
			prev1 =  df.loc[i-1]
			prev2 =  df.loc[i-2]
			fragment = df.iloc[i-window_size:i,:]
			fragment1 = df.iloc[i-window_size:i+1,:]
			max_close = fragment.max().close
			min_close = fragment.min().close

			#### BUY ANALYSIS #####
			#if (symbol == "BCDBTC" and last_row.date=="2019-05-29 13:00"): pdb.set_trace()

			### rolling mean ###
			#buy_rmean_cond =  abs(last_row.rmean48) > abs(prev1.rmean48 * 5) #and prev1.rmean48 > 0
			#pacf_50 = pacf(fragment1.close, nlags=window_size)
			#plot_acf(fragment1.close)
			#plot_pacf(fragment1.close)
			#print(fragment1.close.autocorr(lag=31))
			#print('%f' % diff)
			# buy_ma_cond = True
			# for i, f_row in fragment1.iterrows():
			# 	if f_row.rmean72 > last_row.close or f_row.rmean72 < last_row.open:
			# 		buy_ma_cond = False
			# 		break

			### auto corr ###
			# buy_autoc_cond = True
			# auto_corr = acf(fragment1.close)
			# auto_corr = np.delete(auto_corr,0)
			# for ix in range(len(auto_corr)):
			# 	row = auto_corr[ix]
			# 	if ix > 20 and (row < -0.3 or row > 0.3):
			# 		buy_autoc_cond = False

			### price volume change ###
			#buy_change_cond = last_row.change + prev1.change > 5
			#buy_volume_cond =  last_row.volume > prev1.volume * 10
			#buy_vpt_cond = abs(last_row.vpt) > (abs(prev1.vpt) * 25)

			### rectangle ###
			# buy_rectg_cond = True
			# max_close = fragment.close.max()
			# min_close = fragment.close.min()
			# minmax_diff = max_close - min_close
			# oc_diff = last_row.close - last_row.open
			# xr_diff = last_row.close - prev1.close

			# if symbol == "TNTBTC":
			# 	minmax_diff_coef = 0.00000040
			# 	min_spike_diff = 0.00000010
			# 	max_spike_diff = 0.00000017

			# if symbol == "ONGBTC":
			#  	max_candlesize = 0.00000150
			#  	min_candlesize = 0.00000050
			#  	#minmax_diff_coef = 0.00000100

			# if symbol == "KMDBTC":
			#  	max_candlesize = 0.00000070
			#  	min_candlesize = 0.00000025
			#  	#minmax_diff_coef = 0.00000100

			# if symbol == "ASTBTC":
			#  	last_n_max_size = 0.00000015
			#  	last_min_size = 0.00000030
			#  	last_max_size = 0.00000050

			# if symbol == "BCHBTC":
			#   	last_n_max_size = 0.00010
			#   	last_min_size = 0.00015
			#   	last_max_size = 0.00100

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

			buy_prev_candlesize_cond = True
			last_candlesize = last_row.close - last_row.open
			last_n_max_size = last_candlesize / 4
			last_candlesize_mint = last_row.open - (last_candlesize / 2)
			last_candlesize_maxt = last_row.close + (last_candlesize / 2)

			for iy, rowr in fragment.iterrows():
				candle_size = abs(rowr.close - rowr.open)
				if candle_size > last_n_max_size:
					buy_prev_candlesize_cond = False
					break
				if rowr.open < last_candlesize_mint:
					buy_prev_candlesize_cond = False
					break
				if rowr.close > last_candlesize_maxt:
					buy_prev_candlesize_cond = False
					break


			# buy_last_candlesize = False
			# if last_candlesize > last_min_size and last_candlesize < last_max_size:
			#  	buy_last_candlesize = True

			# last_3_bigger_cond = last_row.close > prev1.close and prev1.close > prev2.close and last_row.close > last_row.open and prev1.close > prev1.open and prev2.close > prev2.open #and (last_row.high - last_row.close) > (last_row.open - last_row.low)
			# max_min_cond = max_close < (last_row.close - ((last_row.close - prev2.open) / 2)) and min_close > (prev2.open - ((last_row.close - prev2.open) / 2))

			last_row_bat = (last_row.high - last_row.close) > (last_candlesize/ 4) and last_row.low == last_row.open

			buy_cond = last_row_bat and buy_prev_candlesize_cond and last_row.change < 3 #and buy_last_candlesize
			sell_cond = last_row.close < prev1.close


			# if symbol == "ARDRBTC":
			#  	minmax_diff_coef = 0.00000050
			# if symbol == "DLTBTC":
			#  	minmax_diff_coef = 0.00000100 # 0.00000056
			# if symbol == "BCDBTC":
			#  	minmax_diff_coef = 0.000010
			# if symbol == "GASBTC":
			#  	minmax_diff_coef = 0.000010
			# if symbol == "KMDBTC":
			#  	minmax_diff_coef = 0.0000020
			# if symbol == "REPBTC":
			#  	minmax_diff_coef = 0.000050
			# if symbol == "NASBTC":
			#  	minmax_diff_coef = 0.0000040
			#buy_rectg_cond = last_row.close > (max_close + minmax_diff) and minmax_diff < minmax_diff_coef
			#buy_rectg_cond = max_close < last_row.close and f_row.rmean72

			#format(minmax_diff_coef, '.8f')
			#format(minmax_diff , '.8f')
			#format(xr_diff, '.8f')

			### detect pattern ###
			# part1 = fragment1.iloc[0:16,:]
			# part2 = fragment1.iloc[16:32,:]
			# part3 = fragment1.iloc[32:48,:]
			# max1 = part1.close.max()
			# min1 = part1.close.min()
			# max2 = part2.close.max()
			# min2 = part2.close.min()
			# max3 = part3.close.max()
			# min3 = part3.close.min()
			# mean1=np.mean([max1,min1])
			# mean2=np.mean([max2,min2])
			# mean3=np.mean([max3,min3])
			# diff1 = max1 - min1
			# diff2 = max2 - min2
			# diff3 = max3 - min3
			# breakout = last_row.close
			# maxmin_cond = max1 > min2 and max1 > min3 and max2 > min1 and max2 > min3 and max3 > min1 and max3 > min2
			# diff_cond = diff1 < minmax_diff_coef and diff2 < minmax_diff_coef and diff3 < minmax_diff_coef
			# breakout_cond = breakout > max1 and breakout > max2 and breakout > max3
			# mean_cond = max2 >= mean1 and max3 >= mean1 and max1 >= mean2 and max3 >= mean2 and max1 >= mean3 and max2 >= mean3
			# pattern_cond = breakout_cond and diff_cond and maxmin_cond and mean_cond

			### rolling mean ###
			#if (symbol == "BCDBTC" and last_row.date=="2019-05-29 13:00"):
			#buy_cond = pattern_cond and buy_vpt_cond and buy_volume_cond and buy_change_cond #buy_rectg_cond and buy_ma_cond  # and buy_autoc_cond

			### candlestick pattern ###
			#buy_cand_pattn = (last_row.inv_hammer == 100)

			# f = df.iloc[i-20:i,:]
			# buy_past = True
			# for iy, rowr in f.iterrows():
			# 	if  rowr['label_knn'] == 1 or rowr['label_cblof'] == 1 or rowr['label_iforest'] == 1:
			# 		buy_past = False
			# 		break


			#buy_cond = (last_row['label_knn'] == 8 and last_row['label_cblof'] == 1 and last_row['label_iforest'] == 1) and last_row.change > 1 and  buy_past == True # last_row.score_knn > 0.1 #
			#sell_cond = (last_row['label_knn'] == 0 and last_row['label_cblof'] == 0 and last_row['label_iforest'] == 0)

			# if last_row.date == "2019-06-22 12:45":
			#   	pdb.set_trace()

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
				#savefig(i,last_row.date,create_ohlc_fragment(i,df,10),"long10",symbol,4)
				#savefig(i,last_row.date,create_ohlc_fragment(i,df,1),"long1",symbol,4)
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
				#print(fragment3)
				#print(sell_type)
				print("buy at: "+str(buy_index))
				print("sell at: "+str(i))
				print("SELL FOR " + sell_type.upper() +" : " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(last_row.date) )
				print("PROFIT: " + str(profit*100))
				print("BALANCE: " + str(balance))
				print("==================================")

				#plt.clf()
				#plt.savefig("./results/plots/"+symbol+"-"+interval+"-"+str(buy_index)+"-"+str(i)+".png")
				#plt.show()
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

def savefig(i,date,ohlc_values,pattern_name,symbol,height):
	plt.cla()
	fig,ax = plt.subplots(figsize = (16,height))
	ax.clear()
	candlestick_ohlc(ax, ohlc_values, width=0.6, colorup='green', colordown='red' )
	ax.axis('off')
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	fn = image_path+symbol+"-"+str(date)+pattern_name+".png"
	plt.savefig(fn, dpi=40)

def create_ohlc_fragment(i,df,forward_window):
	fragment = df.iloc[i-window_size:i+forward_window,:]
	fragment["date"] = fragment.index.values
	ohlc= fragment[['date', 'open', 'high', 'low','close']].copy()
	return ohlc.values


def detect_dtw(df):
	x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
	y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
	euclidean_norm = lambda x, y: np.abs(x - y)
	d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
	print(d)
	plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
	plt.plot(path[0], path[1], 'w')
	plt.show()

def func(x, a, b, c, d):
    return a + b*x - c*np.exp(-d*x)

def plot_fragment(df,path):
	plt.clf()
	df = df.reset_index()
	xdata = df.index.values
	ydata = df.close.values
	popt, pcov = curve_fit(func, xdata, ydata,maxfev=100000)
	plt.figure()
	#plt.plot(xdata, ydata, 'ko', label="Original Noised Data")
	plt.plot(df.close,label="close", marker="o")
	plt.plot(xdata, func(xdata,*popt), 'r-', label="Fitted Curve")
	plt.axis('off')
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	frame1 = plt.gca()
	frame1.axes.xaxis.set_ticklabels([])
	frame1.axes.yaxis.set_ticklabels([])
	plt.savefig(path, dpi=50)

def detect_anomaly(df):
	x_values = df.index.values.reshape(df.index.values.shape[0],1)
	y_values = df.change.values.reshape(df.change.values.shape[0],1)
	clf = KNN()
	clf.fit(y_values)
	clf.predict(y_values)
	df["label_knn"] = clf.predict(y_values)
	df["score_knn"] = clf.decision_function(y_values).round(4)
	clf = IForest()
	clf.fit(y_values)
	clf.predict(y_values)
	df["label_iforest"] = clf.predict(y_values)
	df["score_iforest"] = clf.decision_function(y_values).round(4)
	clf = CBLOF()
	clf.fit(y_values)
	clf.predict(y_values)
	df["label_cblof"] = clf.predict(y_values)
	df["score_cblof"] = clf.decision_function(y_values).round(2)
	return df

def print_df(df):
	with pd.option_context('display.max_rows', None):
		print(df)

if __name__ == '__main__':
    #classify_data()
	#prepare_cnn_ticker()
	backtest()
	#prepare_lstm_data()