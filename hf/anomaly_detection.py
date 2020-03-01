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
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import os
pd.set_option("display.precision", 9)
pd.set_option('display.max_rows', 3000)
pd.options.mode.chained_assignment = None

base_path = "/home/canercak/Desktop/dev/projectlife"
if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
	base_path = "/Users/apple/Desktop/dev/projectlife"
path = base_path +"/data/ticker2/"

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

conditions = [{'name': 'detect trend', 'entry_price': 0, 'action': HOLD, 'trade_count': 0, 'balance': initial_balance, 'buy_mode': True}]

def plot_symbols():
	SYMBOLS = []
	dir = os.listdir(path)
	for s in dir:
		if ".py" not in s and ".DS_Store" not in s:
			SYMBOLS.append(s.split(".csv")[0])
	for symbol in SYMBOLS:
		data_base = read_csv(path+symbol+".csv")
		df = DataFrame(data_base)
		df.columns = ['symbol','date','price_change','price_change_percent','last_price','best_bid_price','best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
		df['qav_sma50'] = df.total_traded_quote_asset_volume.rolling(50).mean()
		df['qav_sma100'] = df.total_traded_quote_asset_volume.rolling(100).mean()
		df['qav_sma200'] = df.total_traded_quote_asset_volume.rolling(200).mean()
		df['last_sma100'] = df.last_price.rolling(100).mean()
		df['last_sma200'] = df.last_price.rolling(200).mean()
		df['last_sma600'] = df.last_price.rolling(600).mean()
		plot_whole(df)

def backtest():
	SYMBOLS =["NAVBTC","KNCBTC","GTOBTC", "REQBTC"]
	# dir = os.listdir(path)
	# for s in dir:
	# 	if ".py" not in s and ".DS_Store" not in s:
	#  		SYMBOLS.append(s.split(".csv")[0])

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
		data_base = read_csv(path+symbol+".csv") #
		df = DataFrame(data_base)
		df.columns = ['symbol','date','price_change','price_change_percent','last_price','best_bid_price','best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
		# df = DataFrame(data_base, columns=['symbol','date','price_change','price_change_percent','last_price','best_bid_price',
		# 		'best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume'])
		#df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
		df['qav_sma50'] = df.total_traded_quote_asset_volume.rolling(50).mean()
		df['qav_sma100'] = df.total_traded_quote_asset_volume.rolling(100).mean()
		df['qav_sma200'] = df.total_traded_quote_asset_volume.rolling(200).mean()
		df['qav_sma400'] = df.total_traded_quote_asset_volume.rolling(400).mean()
		df['last_sma100'] = df.last_price.rolling(100).mean()
		df['last_sma200'] = df.last_price.rolling(200).mean()
		df['last_sma400'] = df.last_price.rolling(400).mean()
		df['last_sma600'] = df.last_price.rolling(600).mean()
		df['last_sma1000'] = df.last_price.rolling(1000).mean()

		# df_x = df
		# df = df.iloc[63931:64932]
		# fragment = detect_anomaly(df)
		# fragment_sum = fragment.groupby(['score_qav', 'label_qav'], as_index=False, sort=False)[[ "change_qav", "change_price"]].sum()
		# print(fragment[['symbol','last_price', 'total_traded_quote_asset_volume', 'label_qav', 'score_qav','change_qav','change_price']].tail(1000))
		# print(fragment_sum)
		# plot_whole(df_x)
		# #pdb.set_trace()

		df = df.reset_index()
		df = df.fillna(0)
		window_size = 1000
		found_dates=[]
		found_list=[]

		for i, row in df.iterrows():
			start_time = time.time()
			current_price = row['last_price']
			current_tick += 1
			if i > window_size:
				fragment = df.iloc[i-window_size:i,:]
				fragment = detect_anomaly(fragment)
				fragment = fragment.reset_index()
				last =  fragment.iloc[-1,:]
				prev1 =  fragment.iloc[-2,:]

				fragment_sum = fragment.groupby(['score_qav', 'label_qav'], as_index=False, sort=False)[[ "change_qav", "change_price"]].sum()
				first_980 = fragment[:980]
				last_20 = fragment[-20:]
				last_20_nodup_score = last_20.drop_duplicates(subset="score_qav")

				conditions[0]['buy_cond'] =(
											(first_980.label_qav == 0).all() and
											#kural1 - ilk 980 tanesi 0 olcak
											len(fragment_sum) >= 3 and
											fragment_sum.label_qav.iloc[0] == 0 and (fragment_sum.label_qav.iloc[-1] == 1 and fragment_sum.label_qav.iloc[-2] == 1) and
											len(search_sequence_numpy(fragment_sum.label_qav.values,np.array([0, 1, 0]))) == 0 and
											len(search_sequence_numpy(fragment_sum.label_qav.values,np.array([1, 0, 1]))) == 0 and
										   	#011  veya benzeri patternlar olcak
										   	(fragment_sum[fragment_sum['label_qav'] == 1].change_qav).sum() > 5 and
										   	#kural3 - 0'ların ve 1'lerin change price toplamıu 3'ten buyuk olcak
										    trendline(last_20[last_20['label_qav'] == 1].total_traded_quote_asset_volume) > 0 #and
										    #kural4 - trendline yuksek olcak
										   )

				conditions[0]['sell_cond'] =(last['last_sma200'] < prev1['last_sma200'])


				for ic, cond in enumerate(conditions):
					if cond['buy_mode'] and cond['buy_cond']:
						printLog("CONDITION " + str(ic+1) +" IS BUYING....")
						conditions[ic]['action'] = BUY
						conditions[ic]['entry_price']  =  current_price
						conditions[ic]['buy_mode'] = False
						printLog("##### TRADE " +  str(cond['trade_count']) + " #####")
						printLog("BUY: " +symbol+" for "+ str(cond['entry_price']) + " at " +  str(last.date) + " - index: " +  str(last['index']))
						# fragment_tmp = df.iloc[i-window_size:i+20,:]
						# fragment_tmp = detect_anomaly(fragment_tmp)
						printLog(fragment[['index','date','symbol','last_price', 'total_traded_quote_asset_volume', 'label_qav', 'score_qav','change_qav','change_price']].tail(100))
						printLog(fragment_sum)
						#printLog("20 after----->>>>>>>")
						#printLog(fragment_tmp[['index','symbol','last_price', 'total_traded_quote_asset_volume', 'label_qav', 'score_qav','change_qav','change_price']].tail(100))
						#pdb.set_trace()
					elif not cond['buy_mode'] and cond['sell_cond']:
						printLog("CONDITION " + str(ic+1) +" IS SELLING....")
						conditions[ic]['action'] = SELL
						exit_price = current_price
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
						#print("DATE BETWEEN "+ symbol +": "+ str(df.head(1).date) + "-"+str(df.tail(1).date))
						printLog("TOTAL BALANCE: "+ str(cond['balance']))
						printLog("TRADE COUNT: "+ str(cond['trade_count']))
					printLog("**********************************")

				if i % 1000 == 0:
					printLog(symbol+"-"+str(row['index']))

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


def detect_trend(df):
	decomposition = seasonal_decompose(df.total_traded_quote_asset_volume, model='multiplicative', extrapolate_trend='freq',  period=100)
	matplotlib.rcParams['figure.figsize'] = [9.0,5.0]
	fig = decomposition.plot()
	trace1 = go.Scatter(
	    x = df.date,y = decomposition.trend,
	    name = 'Trend'
	)
	trace2 = go.Scatter(
	    x = df.date,y = decomposition.seasonal,
	    name = 'Seasonal'
	)
	trace3 = go.Scatter(
	    x = df.date,y = decomposition.resid,
	    name = 'Residual'
	)
	trace4 = go.Scatter(
	    x = df.date,y = df.total_traded_quote_asset_volume,
	    name = 'Mean Stock Value'
	)
	plt.show()

def trendline(data, order=1):
    coeffs = np.polyfit(data.index.values, list(data), order)
    slope = coeffs[-2]
    return float(slope)

def plot_whole(df):
	plt.clf()
	fig, axes = plt.subplots(nrows=2, ncols=1)
	df.total_traded_quote_asset_volume.plot(ax=axes[0] , color="blue", style='.-')

	df.qav_sma50.plot(ax=axes[0], color="red")
	df.qav_sma100.plot(ax=axes[0], color="orange")
	df.qav_sma200.plot(ax=axes[0], color="brown")

	df.last_price.plot(ax=axes[1], style='.-')
	df.last_sma100.plot(ax=axes[1], color="yellow")
	df.last_sma200.plot(ax=axes[1], color="purple")
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
	with open(base_path+'/logs/output.out','a') as file:
		print(*args, **kwargs, file=file)


if __name__ == '__main__':
	#plot_symbols()
	backtest()


