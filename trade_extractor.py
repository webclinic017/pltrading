from math import sqrt
from dateutil import parser
from configparser import ConfigParser
from pandas import Series
import matplotlib.dates as mdates
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
import time
import statistics
from pandas import datetime
import platform
#import trendet
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import matplotlib
if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
	matplotlib.use("macOSX")
#from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np
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
import math
import numpy as np
import pandas
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm, mstats
from datetime import datetime, timedelta
import os
pd.set_option("display.precision", 9)
pd.set_option('display.max_rows', 3000)
pd.options.mode.chained_assignment = None

base_path = "/home/canercak/Desktop/dev/projectlife"
if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
	base_path = "/Users/apple/Desktop/dev/projectlife"
path = base_path +"/data/ticker3/"

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

conditions = [{'name': 'detect spike1', 'entry_price': 0, 'action': HOLD, 'trade_count': 0, 'balance': initial_balance, 'buy_mode': True}]

def backtest():
	with open('/Users/apple/Desktop/dev/projectlife/hf/patterns/spike_patterns.json') as json_file:
		patterns = json.load(json_file)
		for pattern in patterns:
			data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/" +  pattern['data'] +"/"+pattern['symbol']+".csv")
			df = pd.DataFrame(data_base)
			symbol = pattern['symbol']
			df.columns = ['symbol','date','price_change','price_change_percent','last_price','best_bid_price','best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
			df['qav_sma50'] = df.total_traded_quote_asset_volume.rolling(50).mean()
			df['qav_sma100'] = df.total_traded_quote_asset_volume.rolling(100).mean()
			df['qav_sma200'] = df.total_traded_quote_asset_volume.rolling(200).mean()
			df['last_sma100'] = df.last_price.rolling(100).mean()
			df['last_sma200'] = df.last_price.rolling(200).mean()
			df['last_sma600'] = df.last_price.rolling(600).mean()

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
			window_size = 2000

			df_x = df
			df = df.iloc[pattern['end'] - window_size-100:pattern['end']+2000]

			df = df.reset_index()
			df = df.fillna(0)

			for i, row in df.iterrows():
				start_time = time.time()
				current_price = row['last_price']
				current_ask_price = row['best_ask_price']
				current_bid_price = row['best_bid_price']
				current_tick += 1
				if i > window_size:
					fragment = df.iloc[i-window_size:i,:]
					fragment = detect_anomaly(fragment)
					fragment = fragment.reset_index()
					last =  fragment.iloc[-1,:]
					prev1 =  fragment.iloc[-2,:]

					fragment_sum = fragment.groupby(['score_qav', 'label_qav'], as_index=False, sort=False)[[ "change_qav", "change_price"]].sum()
					first_1980 = fragment[:1980]
					last_20 = fragment[-20:]
					last_20_nodup_score = last_20.drop_duplicates(subset="score_qav")

					conditions[0]['buy_cond'] =(
												(first_1980.label_qav == 0).all() and
												#kural1 - geriye 2000 bakacaksın. 1000 inanılmaz kayıp veriyor. sakın 1000'e donme.
												len(fragment_sum) >= 2 and
												(fragment_sum.label_qav.iloc[0] == 0 and fragment_sum.label_qav.iloc[-1] == 1  ) and
												len(search_sequence_numpy(fragment_sum.label_qav.values,np.array([0, 1, 0]))) == 0 and
												len(search_sequence_numpy(fragment_sum.label_qav.values,np.array([1, 0, 1]))) == 0 and
												len(search_sequence_numpy(fragment_sum.label_qav.values,np.array([1, 0, 0, 1]))) == 0 and
												len(search_sequence_numpy(fragment_sum.label_qav.values,np.array([1, 0, 0, 1, 1]))) == 0 and
												len(search_sequence_numpy(fragment_sum.label_qav.values,np.array([1, 0, 0, 0, 1, 1]))) == 0 and
											   	#kural2 - en az 011 veya fazlası patternlar olcak. Kesinlikle 01 olamaz. Kesinlikle 01 olur obur turlu geç alıp kaybediyor. Onemlş lan arkadan nasıl geldiği
											   	(fragment_sum[fragment_sum['label_qav'] == 1].change_qav).sum() > 5 and
											   	#kural3 - 1'lerin change_qav toplamıu 4'ten buyuk olcak. Onceden 5'ti onemli yerleri kacırıyor. 4 olamaz. 5ten bile fazla olması lazım
											    # trendline(last_20[last_20['label_qav'] == 1].total_traded_quote_asset_volume) > 0 and
											    # #kural4 - trendline yuksek olcak
											    (fragment_sum[fragment_sum['label_qav'] == 1].change_price).sum() > 0 and
											    (fragment_sum[fragment_sum['label_qav'] == 1].change_price).sum() < 10 and
											    #kural5 - 1'lerin change_price toplamıu 0.5'den buyuk 10'dan kucuk olcak. 1 dedin geç aldı 0.5 dedin geç aldı en iyisi 0. onemli olan change_qav
											    fragment_sum[fragment_sum['label_qav'] == 1].change_qav.max() > fragment_sum[fragment_sum['label_qav'] == 0].change_qav.max()
											    #kural6 - onemli. ticker2storj188785 ticker3sys125169 gibi durumlardan korunmak için.
											    #(not fragment_sum[fragment_sum['label_qav'] == 1].change_qav.is_monotonic_decreasing)
											    #kural7 - change_qav azalıyor olmicak. Bu kuralı kapatıyorum. sortun yaratıyor. ticker3dnt14569 gibi durumlar oluyor.
											   )

					conditions[0]['sell_cond'] =(last['last_sma100'] < prev1['last_sma100'])
					#spike için en uygunu 100luk test ettim onayladım. Hep daha fazal kazanıyor hem aha az kaybediyor.

					for ic, cond in enumerate(conditions):
						if cond['buy_mode'] and cond['buy_cond']:
							conditions[ic]['action'] = BUY
							conditions[ic]['entry_price']  =  current_ask_price
							conditions[ic]['buy_mode'] = False
							if ic ==0:
								printLog("PATTERN: "+ pattern['type'])
								printLog("--------------------------")
								printLog("CONDITION " + str(ic+1) +" IS BUYING....")
								printLog("##### TRADE " +  str(cond['trade_count']) + " #####")
								printLog("BUY: " +symbol+" for "+ str(cond['entry_price']) + " at " +  str(last.date) + " - index: " +  str(last['index']))
								printLog(fragment[['index','date','symbol','last_price', 'total_traded_quote_asset_volume', 'label_qav', 'score_qav','change_qav','change_price']].tail(100))
								printLog(fragment_sum)
						elif not cond['buy_mode'] and cond['sell_cond']:
							printLog("CONDITION " + str(ic+1) +" IS SELLING....")
							conditions[ic]['action'] = SELL
							exit_price =  current_bid_price
							profit = ((exit_price - cond['entry_price'])/cond['entry_price'] + 1)*(1-transaction_fee)**2 - 1
							conditions[ic]['balance'] = conditions[ic]['balance'] * (1.0 + profit)
							conditions[ic]['trade_count'] += 1
							conditions[ic]['buy_mode'] = True
							printLog("SELL: " + symbol+" for "+ str(exit_price) + " at " +  str(last.date) + " - index: " +  str(last['index']))
							printLog("PROFIT: " + str(profit*100))
							printLog("BALANCE: " + str(cond['balance']))
						else:
							conditions[ic]['action'] = HOLD

					if (current_tick > len(df)-1):
						printLog("*********TOTAL RESULTS*************************")
						for ic, cond in enumerate(conditions):
							printLog("SYMBOL: "+ symbol)
							printLog("CONDITION NUMBER: "+ str(ic))
							printLog("TOTAL BALANCE: "+ str(cond['balance']))
							printLog("TRADE COUNT: "+ str(cond['trade_count']))
						printLog("**********************************")

def trendline(data, order=1):
	coeffs = np.polyfit(data.index.values, list(data), order)
	slope = coeffs[-2]
	return float(slope)

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

def search_sequence_numpy(arr,seq):
	# Store sizes of input array and sequence
	Na, Nseq = arr.size, seq.size

	# Range of sequence
	r_seq = np.arange(Nseq)

	# Create a 2D array of sliding indices across the entire length of input array.
	# Match up with the input sequence & get the matching starting indices.
	M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

	# Get the range of those indices as final output
	if M.any() >0:
		return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
	else:
		return []         # No match found


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

def printLog(*args, **kwargs):
	print(*args, **kwargs)
	with open(base_path+'/logs/trade_log.txt','a') as file:
		print(*args, **kwargs, file=file)


if __name__ == '__main__':
	backtest()

