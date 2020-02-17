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
	SYMBOLS = ["BTS/BTC"] #ICX #ENJ
	for symbol in SYMBOLS:
		symbol =  symbol.split("/")[0] + symbol.split("/")[1]
		data_base = read_csv("/home/canercak/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
		df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		df['sma7'] = df.close.rolling(7).mean()
		df['sma25'] = df.close.rolling(25).mean()
		df['sma99'] = df.close.rolling(99).mean()
		df = df.drop(["low"], axis=1) 
		df['change'] = df.close.pct_change(periods=1)*100
		df = df.fillna(0)

		#df = df.iloc[71200:71880, : ]
		#plot_fragment(df)
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

				# buy_cond = (last['label_volume'] == 1  and ## yapmaya çalıştıpın buna cevap vermek: https://prnt.sc/qyxfk0
				#  	        last['label_close'] == 1 and
				#  	        last['change'] > 0 and
				#  	        last['volume'] > prev1['volume'] * 3 and
				#  	        sum(fragment.tail(10)['label_close'].astype("str").str.contains("0")) == 9 and
				#  	        sum(fragment.tail(10)['label_volume'].astype("str").str.contains("0")) == 9 )

				buy_cond = (last['label_volume'] == 1  and ## yapmaya çalıştıpın buna cevap vermek: https://prnt.sc/qyxfk0
				 	        prev1['label_volume'] == 1 and
				 	        prev1['change'] < -2 and
				 	        last['change'] > 1 and
				 	        #prev1['volume'] > 0 and
				 	        #last['sma25'] > last['sma99'] and #last['sma7'] > last['sma25'] and 
				 	        #prev1['sma7'] < last['sma7'] and prev1['sma25'] < last['sma25'] and prev1['sma99'] < last['sma99'] and
				 	        #last['volume'] > prev1['volume'] * 50 and
				 	        sum(fragment.tail(20)['label_close'].astype("str").str.contains("0")) > 18 and
				 	        sum(fragment.tail(20)['label_volume'].astype("str").str.contains("0")) == 18 )

				print("short algo - "+symbol + " - "+ last['date'])
				if buy_cond:
					if last['date'] not in found_dates:
						found_dates.append(last['date'])
						found_list.append({last['date']: fragment.tail(200)})

		print(found_dates)
		print(found_list)




def plot_fragment(fragment):
	plt.clf()
	fragment = fragment.reset_index()
	plt.figure()
	plt.plot(fragment.close,label="close", marker="o")
	plt.axis('off')
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	frame1 = plt.gca()
	frame1.axes.xaxis.set_ticklabels([])
	frame1.axes.yaxis.set_ticklabels([])
	pdb.set_trace()

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


# #########
# ---------
# 2020-01-23 20:06 enj ******* hiçbirşey bundan kolay degil
# 71811  2020-01-23 20:51  0.00001024  0.00001027  0.00001027    11427  ...  0.29296875            0 -19.63108704             0    3.32190208
# 71812  2020-01-23 20:52  0.00001025  0.00001031  0.00001028   130141  ...  0.09737098            0 -19.63108704             0    3.32190208
# 71813  2020-01-23 20:53  0.00001028  0.00001031  0.00001029   229301  ...  0.09727626            0 -19.63108704             0    3.32190208
# 71814  2020-01-23 20:54  0.00001029  0.00001035  0.00001029   228124  ...  0.00000000            0 -19.63108704             0    3.32190208
# 71815  2020-01-23 20:55  0.00001029  0.00001033  0.00001029    45234  ...  0.00000000            0 -19.63108704             0    3.32190208
# 71816  2020-01-23 20:56  0.00001030  0.00001032  0.00001030     4441  ...  0.09718173            0 -20.47015074             0    3.32190208
# 71817  2020-01-23 20:57  0.00001032  0.00001037  0.00001036    92311  ...  0.58252427            0 -20.47015074             0    3.32190208
# 71818  2020-01-23 20:58  0.00001035  0.00001037  0.00001036    54697  ...  0.00000000            0 -20.47015074             0    3.32190208
# 71819  2020-01-23 20:59  0.00001037  0.00001040  0.00001040    74784  ...  0.38610039            0 -20.47015074             0    3.32190208
# 71820  2020-01-23 21:00  0.00001040  0.00001042  0.00001034   129053  ... -0.57692308            0 -20.47015074             0    3.32190208
# 71821  2020-01-23 21:01  0.00001035  0.00001035  0.00000958  4806543  ... -7.35009671            0 -19.36805267             1    3.32192806
# 71822  2020-01-23 21:02  0.00000963  0.00001011  0.00000985  2588310  ...  2.81837161            0 -20.56549861             1    3.32192806
# 71823  2020-01-23 21:03  0.00000992  0.00001098  0.00001064  5406143  ...  8.02030457            1 -15.13923776             1    3.32192806
# 71824  2020-01-23 21:04  0.00001064  0.00001091  0.00001085  2000549  ...  1.97368421            1 -14.72420159             1    3.32192806
# 71825  2020-01-23 21:05  0.00001084  0.00001093  0.00001065   936929  ... -1.84331797            1 -15.13923776             1    3.32192758
# 71826  2020-01-23 21:06  0.00001069  0.00001074  0.00001074   903482  ...  0.84507042            1 -15.13923776             1    3.32192758
# 71827  2020-01-23 21:07  0.00001072  0.00001095  0.00001095   900009  ...  1.95530726            1 -14.72420159             1    3.32192758
# 71828  2020-01-23 21:08  0.00001094  0.00001094  0.00001082   538001  ... -1.18721461            1 -14.72420159             0    3.32190208
# 71829  2020-01-23 21:09  0.00001086  0.00001092  0.00001071   792400  ... -1.01663586            1 -15.13923776             1    3.32192758
# 71830  2020-01-23 21:10  0.00001075  0.00001076  0.00001046   798459  ... -2.33426704            0 -20.47015074             1    3.32192758
# 71831  2020-01-23 21:11  0.00001048  0.00001050  0.00001037  1056422  ... -0.86042065            0 -20.47015074             1    3.32192758
# 71832  2020-01-23 21:12  0.00001035  0.00001059  0.00001058   681531  ...  2.02507232            0 -18.72419660             1    3.32192758
# 71833  2020-01-23 21:13  0.00001058  0.00001060  0.00001039   419705  ... -1.79584121            0 -20.47015074             0    3.32190208
# 71834  2020-01-23 21:14  0.00001040  0.00001050  0.00001049   467313  ...  0.96246391            0 -18.72419660             0    3.32190208
# 71835  2020-01-23 21:15  0.00001045  0.00001050  0.00001044   202016  ... -0.47664442            0 -20.47015074             0    3.32190208
# 71836  2020-01-23 21:16  0.00001041  0.00001050  0.00001037   500212  ... -0.67049808            0 -20.47015074             0    3.32190208
# 71837  2020-01-23 21:17  0.00001036  0.00001038  0.00001034    75086  ... -0.28929605            0 -20.47015074             0    3.32190208
# 71838  2020-01-23 21:18  0.00001034  0.00001044  0.00001041   836716  ...  0.67698259            0 -20.47015074             1    3.32192758
# 71839  2020-01-23 21:19  0.00001044  0.00001046  0.00001042   133726  ...  0.09606148            0 -20.47015074             0    3.32190208
# 71840  2020-01-23 21:20  0.00001042  0.00001045  0.00001041   152840  ... -0.09596929            0 -20.47015074             0    3.32190208

