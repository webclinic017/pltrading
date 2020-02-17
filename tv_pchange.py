


from pyculiarity import detect_ts
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
from pyod.models.knn import KNN
pd.set_option("display.precision", 8)
pd.set_option('display.max_rows', 1500)

datatype ="local"
transaction_fee = 0.00125
initial_balance = 100
BUY, SELL, HOLD = 0, 1, 2
results = {}
interval = "1h"
config = ConfigParser()
config.read("config.ini")
exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})
window_size = 48
profit_perc = 1.11
stoploss_perc = 0.98

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
		SYMBOLS = [
            "NAS/BTC", "ARDR/BTC", "GXS/BTC", "BCD/BTC",
            "HC/BTC","TNT/BTC", "BTG/BTC", "ATOM/BTC",
            "WAN/BTC","EOS/BTC","SKY/BTC","RDN/BTC",
            "LTC/BTC","REP/BTC","QKC/BTC", "IOTA/BTC",
            "ZEC/BTC", "DLT/BTC", "ENJ/BTC","GO/BTC",
            "LINK/BTC","EDO/BTC","GRS/BTC","APPC/BTC",
            "ARK/BTC","NULS/BTC","SNGLS/BTC","WAVES/BTC"
        ]
		for symbol in SYMBOLS:
			if datatype == "remote":
				data_base = exchange.fetch_ohlcv(symbol, interval,limit=1000)
				df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
				df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
				symbol =  symbol.split("/")[0] + symbol.split("/")[1]
			elif datatype =="local":
				symbol =  symbol.split("/")[0] + symbol.split("/")[1]
				data_base = read_csv("/Users/apple/Desktop/dev/projectlife/data/full/Binance_"+symbol+"-"+interval+".csv")
				df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
			evaluate_symbol(df,symbol,SYMBOLS)

def evaluate_symbol(df,symbol,SYMBOLS):
	df.set_index('date')
	df['change'] = df.close.pct_change(periods=1)*100
	#df["rmean24"] = df.change.rolling(24).mean()
	df["rmean48"] = df.change.rolling(48).mean()
	#df["rmean72"] = df.change.rolling(72).mean()
	#df["cmean24"] = df.close.rolling(24).mean()
	#df["cmean48"] = df.close.rolling(48).mean()
	#df["cmean72"] = df.close.rolling(72).mean()
	df['vpt'] = ta.volume.volume_price_trend(df["close"], df["volume"], fillna=False).round(4)
	#df["vpt_change"] =  (df.vpt.pct_change(periods=1)*100).round(4)
	df['sma25'] = talib.SMA(df['close'],25)
	df['sma100'] = talib.SMA(df['close'],100)
	#df["max72"] = df.vpt.rolling(72).max().round(4)
	#df["min72"] = df.vpt.rolling(72).min().round(4)
	#pdb.set_trace()
	#df["maxmin_diff"] = df["max72"] - df["min72"]
	#df["maxmin_mean"] = (df["min72"] + df["min72"]) / 2
	#df = df.drop(['high',"low"], axis=1)
	df = df.fillna(0)
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
		fragment1 = df.iloc[i-window_size:i,:]
		fragment2 = df.iloc[i-window_size:i+1,:]
		fragment3 = df.iloc[i-window_size:i+10,:]
		if i > window_size:
			#df = detect_anomaly(df)
			#df = detect_dtw(df)
			last_row =  df.loc[i]
			prev3 =  df.loc[i-3]
			prev2 =  df.loc[i-2]
			prev1 =  df.loc[i-1]

			#### FIND CONSOLIDATION #####
			#if symbol == "HCBTC": "2019-05-30 03:00":
			#values = fragment.maxmin_mean.values

			#unique = np.unique(values)
			#occurances = collections.Counter(values)

			#if (symbol == "BCDBTC" and last_row.date=="2019-05-29 13:00"): pdb.set_trace()
			#if (symbol == "TNTBTC" and last_row.date=="2019-05-21 16:00"): pdb.set_trace()
			#if (symbol == "HCBTC" and last_row.date=="2019-05-30 03:00"): pdb.set_trace()
			#if (symbol == "GXSBTC" and last_row.date=="2019-06-04 11:00"): pdb.set_trace()

			#### BUY CONDITIONS #####

			### rolling mean ###
			#buy_rmean_cond =  abs(last_row.rmean48) > abs(prev1.rmean48 * 5) #and prev1.rmean48 > 0
			#pacf_50 = pacf(fragment1.close, nlags=window_size)
			#plot_acf(fragment1.close)
			#plot_pacf(fragment1.close)
			#print(fragment1.close.autocorr(lag=31))
			#print('%f' % diff)
			# buy_ma_cond = True
			# for i, f_row in fragment1.iterrows():
			# 	if f_row.sma100 > max_close:
			# 		buy_ma_cond = False

			### auto corr ###
			# buy_autoc_cond = True
			# auto_corr = acf(fragment1.close)
			# auto_corr = np.delete(auto_corr,0)
			# for ix in range(len(auto_corr)):
			# 	row = auto_corr[ix]
			# 	if ix > 20 and (row < -0.3 or row > 0.3):
			# 		buy_autoc_cond = False

			### price volume change ###
			buy_change_cond = last_row.change + prev1.change > 5
			buy_volume_cond =  last_row.volume > prev1.volume * 10
			buy_vpt_cond = abs(last_row.vpt) > (abs(prev1.vpt) * 25)

			### rectangle ###
			# buy_rectg_cond = False
			# max_close = fragment1.close.max()
			# min_close = fragment1.close.min()
			# minmax_diff = max_close - min_close
			# oc_diff = last_row.close - last_row.open
			# # for iy, row in fragment1.iterrows():
			# # 	if row.close >
			# # 		buy_rectg_cond = True
			if symbol == "GXSBTC":
				minmax_diff_coef = 0.000006
			if symbol == "TNTBTC":
				minmax_diff_coef = 0.00000025
			if symbol == "BCDBTC":
				minmax_diff_coef = 0.000010
			if symbol == "HCBTC":
				minmax_diff_coef = 0.0000065
			# buy_rectg_cond = last_row.close > (max_close + minmax_diff) and minmax_diff < minmax_diff_coef

			### detect pattern ###
			part1 = fragment1.iloc[0:16,:]
			part2 = fragment1.iloc[16:32,:]
			part3 = fragment1.iloc[32:48,:]
			max1 = part1.close.max()
			min1 = part1.close.min()
			max2 = part2.close.max()
			min2 = part2.close.min()
			max3 = part3.close.max()
			min3 = part3.close.min()
			mean1=np.mean([max1,min1])
			mean2=np.mean([max2,min2])
			mean3=np.mean([max3,min3])
			diff1 = max1 - min1
			diff2 = max2 - min2
			diff3 = max3 - min3
			breakout = last_row.close

			maxmin_cond = max1 > min2 and max1 > min3 and max2 > min1 and max2 > min3 and max3 > min1 and max3 > min2
			diff_cond = diff1 < minmax_diff_coef and diff2 < minmax_diff_coef and diff3 < minmax_diff_coef
			breakout_cond = breakout > max1 and breakout > max2 and breakout > max3
			mean_cond = max2 >= mean1 and max3 >= mean1 and max1 >= mean2 and max3 >= mean2 and max1 >= mean3 and max2 >= mean3
			pattern_cond = breakout_cond and diff_cond and maxmin_cond and mean_cond

			if (symbol == "BCDBTC" and last_row.date=="2019-05-29 13:00"): pdb.set_trace()

			buy_cond = pattern_cond and buy_vpt_cond and buy_volume_cond and buy_change_cond #buy_rectg_cond and buy_ma_cond  # and buy_autoc_cond

			#### SELL CONDITIONS #####
			sell_change_cond = (last_row.change < prev1.change)
			sell_cond = sell_change_cond
			stopl_cond = False

			if buy_cond and buy_mode:
				buy_index = i
				action = BUY
				entry_price =  current_price
				entry_tick = current_tick
				quantity = balance / entry_price
				buy_mode = False
				print("##### TRADE " +  str(trade_count) + " #####")
				print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(last_row.date))
				#plot_fragment(df,symbol)
				pdb.set_trace()
			elif not buy_mode and (sell_cond or stopl_cond):
				#pdb.set_trace()
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
				print("PROFIT: " + str(profit))
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

	# ax =SYMBOLS[-1]
	# if (symbol == ax.split("/")[0]+ax.split("/")[1]):
	# 	print("#########FINAL BALANCES#####################")
	# 	final_balance = 0
	# 	for symbol in SYMBOLS:
	# 		balance = results[symbol]['balance'][0]
	# 		print(symbol + ": "+str(balance))
	# 		final_balance +=balance
	# 	print("============================================")
	# 	print("Trade count: " + str(trade_count))
	# 	print("Win count: " + str(win_count))
	# 	print("Loss count: " + str(loss_count))
	# 	print("FINAL BALANCE: " + str(final_balance))


def detect_anomaly(df):
	clf = KNN()
	x_values = df.change.values.reshape(df.index.values.shape[0],1)
	y_values = df.change.values.reshape(df.change.values.shape[0],1)
	clf.fit(y_values)
	clf.predict(y_values)
	df["out_label"] = clf.predict(y_values).round(4)  #fit_predict_score
	df["out_score"] = clf.decision_function(y_values).round(4)
	return df

def detect_dtw(df):
	pdb.set_trace()
	x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
	y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
	euclidean_norm = lambda x, y: np.abs(x - y)
	d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
	print(d)
	plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
	plt.plot(path[0], path[1], 'w')
	plt.show()

def plot_fragment(df,symbol):
	fragment = df#.tail(200)
	plt.clf()
	plt.title(symbol + " " +  str(fragment.iloc[-1].date))
	plt.plot(fragment.change, marker="o", label="change",figsize=(30,10))
	plt.plot(fragment.rmean24, marker="o", label="rmean24",figsize=(30,10))
	plt.plot(fragment.rmean48, marker="o", label="rmean48",figsize=(30,10))
	plt.plot(fragment.rmean64, marker="o", label="rmean64",figsize=(30,10))
	plt.legend()
	print(fragment)
	#plt.savefig("./results/plots/"+symbol+"-"+interval+".png")
	plt.show()
	plt.clf()
	plt.plot(fragment.close, marker="o", label="close",figsize=(30,10))
	plt.plot(fragment.cmean24, marker="o", label="cmean24",figsize=(30,10))
	plt.plot(fragment.cmean48, marker="o", label="cmean48",figsize=(30,10))
	plt.legend()
	plt.show()

def classify_data_as_labels():
	with open('/Users/apple/Desktop/dev/projectlife/data/class/patterns.json') as json_file:
		data = json.load(json_file)
		df_array = []
		for elem in data:
			symbol = elem['symbol']
			pattern = elem['patterns']['hunter']
			data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/full/Binance_"+symbol+"-"+"30m"+".csv")
			df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
			df['pattern'] = "none"
			df['symbol'] = symbol
			for i, row in df.iterrows():
				for p in pattern:
					start = pd.to_datetime(p["start"])
					end = pd.to_datetime(p["end"])
					row_date = pd.to_datetime(row.date)
					if start <= row_date <= end:
						df['pattern'][i] = "hunter"
			df_array.append(df)
		df_full = pd.concat(df_array)
		df_full = df_full.reset_index()
		pdb.set_trace()




	# if datatype == "all":
	# 	exchange.load_markets()
	# 	SYMBOLS = exchange.symbols
	# 	save_symbols(SYMBOLS)
	# 	for symbol in SYMBOLS:
	# 		symbol = symbol.split("/")[0] + symbol.split("/")[1]
	# 		df = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/allin/Binance_"+symbol+"-"+interval+".csv")
	# 		df = df.tail(144)
	# 		evaluate_symbol(df,symbol,SYMBOLS)

if __name__ == '__main__':
	classify_data()
	#backtest()