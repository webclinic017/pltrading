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
config = ConfigParser()
config.read("config.ini")
#exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})
profit_perc = 1.11
stoploss_perc = 0.98
dateformat_save = '%Y-%m-%d-%H-%M'

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

def plot_fragment(df,path):
	#plt.clf()
	plt.close('all')
	df = df.reset_index()
	xdata = df.index.values
	ydata = df.close.values
	#popt, pcov = curve_fit(func, xdata, ydata,maxfev=100000)
	plt.figure()
	#plt.plot(xdata, ydata, 'ko', label="Original Noised Data")
	plt.plot(df.close,label="close")# marker="o"
	#plt.plot(xdata, func(xdata,*popt), 'r-', label="Fitted Curve")
	plt.axis('off')
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	frame1 = plt.gca()
	frame1.axes.xaxis.set_ticklabels([])
	frame1.axes.yaxis.set_ticklabels([])
	plt.savefig(path, dpi=50)

def prepare_cnn_patterns(type):
	with open('/Users/apple/Desktop/dev/projectlife/utils/classification/all_patterns.json') as json_file:
		data = json.load(json_file)
		df_array = []
		interval = "5m"
		window_size = 72
		for elem in data:
			symbol = elem['symbol']
			try:
				#if symbol == "NEOBTC":
				patterns = elem['patterns'][interval]
				data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
				df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
				for i, last_row in df.iterrows():
					if i > window_size:
						for pt in patterns:
							if last_row.date == pt['end']: #  and pt['type'] == "hunter"
								fragment = df.iloc[i-window_size:i,:]
								fragment1 = df.iloc[i-window_size:i+1,:]
								candles = []
								for iy, rowr in fragment.iterrows():
									if rowr.close > rowr.open:
										candle_size = rowr.close - rowr.open
										candles.append(candle_size)
								last_candlesize = last_row.close - last_row.open
								prevn_max_candlesize = max(candles)
								prevn_min_candlesize = min(candles)
								fragment1["date"] = fragment1.index.values
								ohlc= fragment1[['date', 'open', 'high', 'low','close']].copy().values
								path = "./data/images/candle_patterns/"+pt['type'] +"/"+symbol+"-"+str(pt['certainty'])+"-"+last_row.date+".png"
								save_candles(ohlc,path)
			except:
			 	print(interval + " " + symbol +" pattern not found")

def prepare_cnn_ticker(type):
		symbol = "NEOBTC"
		interval = "5m"
		window_size = 72
		data_start_date = pd.to_datetime("2019-02-23 15:55")
		data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
		df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		for i, last_row in df.iterrows():
			last_date = pd.to_datetime(last_row.date)
			if i > window_size and last_date > data_start_date:
				fragment1 = df.iloc[i-window_size:i+1,:]
				if type == "line":
					path = "./data/images/analysis/backtest/"+symbol+"-"+last_row.date+".png"
					plot_fragment(fragment1, path)
				else:
					fragment1["date"] = fragment1.index.values
					ohlc= fragment1[['date', 'open', 'high', 'low','close']].copy().values
					path = "./data/images/candle_tickers/"+symbol+"-"+last_row.date+".png"
					print(path)
					save_candles(ohlc,path)

def save_candles(ohlc_values,path):
	plt.close('all')#plt.clf()
	fig,ax = plt.subplots(figsize = (16,8))
	ax.clear()
	candlestick_ohlc(ax, ohlc_values, width=0.6, colorup='green', colordown='red' )
	ax.axis('off')
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	plt.savefig(path, dpi=30)


def prepare_UCR_data():
	with open('/Users/apple/Desktop/dev/projectlife/utils/classification/found_patterns.json') as json_file:
		data = json.load(json_file)
		df_array = []
		interval = "5m"
		window_size = 72
		X_train, Y_train,  X_test, Y_test = [],[],[],[]
		train_start_date = pd.to_datetime("2018-01-01 00:00")
		df_final = pd.DataFrame()
		for elem in data:
			try:
				symbol = elem['symbol']
				patterns = elem['patterns'][interval]
				data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
				df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
				df = df.drop(['high',"low","volume"], axis=1)
				train_patterns, test_patterns = [],[]
				for p in patterns:
					end = pd.to_datetime(p["end"])
					if end > train_start_date and p["type"] == "hunter":
						train_patterns.append(p["end"])
				for i, last_row in df.iterrows():
					row_date = pd.to_datetime(last_row.date)
					if symbol == "BATBTC" and last_row.date =="2018-05-04 00:55":
						continue
					if symbol == "ONTBTC" and last_row.date =="2019-03-08 16:15":
						continue
					else:
						if i > window_size and row_date > train_start_date:
							middle_slide = df.iloc[i-window_size:i+1,:]
							middle_close = middle_slide['close'].values.tolist()
							if last_row.date in train_patterns:
								middle_close.insert(0, "6")
								df_final = df_final.append(pd.Series(middle_close),ignore_index=True)
								path = "./data/images/analysis/patterns/"+symbol+"-"+last_row.date+".png"
								plot_fragment(middle_slide,path)
							else:
								if i % 5000 == 0:
									middle_close.insert(0, "9")
									df_final = df_final.append(pd.Series(middle_close),ignore_index=True)
									path = "./data/images/analysis/ticker/"+symbol+"-"+last_row.date+".png"
									plot_fragment(middle_slide,path)
			except:
				print(symbol +" pattern not found")
		train, test = train_test_split(df_final, test_size=0.3,shuffle=False)
		train.to_dense().to_csv("/Users/apple/Desktop/dev/projectlife/data/UCR/Projectlife/Projectlife_TRAIN", index = False, header=False, sep=",", encoding='utf-8')
		test.to_dense().to_csv("/Users/apple/Desktop/dev/projectlife/data/UCR/Projectlife/Projectlife_TEST", index = False, header=False, sep=",", encoding='utf-8')

def prepare_UCR_data_full():
	print("prepare_lstm_data_full")
	with open('/Users/apple/Desktop/dev/projectlife/utils/classification/all_patterns.json') as json_file:
		data = json.load(json_file)
		df_array = []
		interval = "5m"
		window_size = 72
		X_train, Y_train,  X_test, Y_test = [],[],[],[]
		train_start_date = pd.to_datetime("2018-05-08 21:10")
		df_final = pd.DataFrame()
		for elem in data:
			try:
				symbol = elem['symbol']
				print(symbol)
				patterns = elem['patterns'][interval]
				data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
				df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
				df = df.drop(['high',"low","volume"], axis=1)
				found_patterns, none_patterns = [],[]
				for p in patterns:
					end = pd.to_datetime(p["end"])
					if end > train_start_date and p["type"] == "hunter":
						found_patterns.append(p["end"])
					if end > train_start_date and p["type"] == "none":
						none_patterns.append(p["end"])
				for i, last_row in df.iterrows():
					row_date = pd.to_datetime(last_row.date)
					if i > window_size and row_date > train_start_date:
						middle_slide = df.iloc[i-window_size:i+1,:]
						middle_close = middle_slide['close'].values.tolist()
						if last_row.date in found_patterns:
							middle_close.insert(0, "1")
						elif last_row.date in none_patterns:
							middle_close.insert(0, "2")
						df_final = df_final.append(pd.Series(middle_close),ignore_index=True)
						#path = "/Users/apple/Desktop/dev/projectlife/data/images/analysis/patterns/"+symbol+"-"+last_row.date+".png"
						#plot_fragment(middle_slide,path)
			except:
			 	print(symbol +" pattern not found")
		train, test = train_test_split(df_final, test_size=0.5,shuffle=False)
		train.to_dense().to_csv("/Users/apple/Desktop/dev/projectlife/data/UCR/Projectlife/Projectlife_TRAIN", index = False, header=False, sep=",", encoding='utf-8')
		test.to_dense().to_csv("/Users/apple/Desktop/dev/projectlife/data/UCR/Projectlife/Projectlife_TEST", index = False, header=False, sep=",", encoding='utf-8')

def prepare_HAR_data():
	direc = '/Users/apple/Desktop/dev/projectlife/data/UCR'
	datadir = direc + '/' + 'Projectlife' + '/' + 'Projectlife'
	data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
	data_test = np.loadtxt(datadir+'_TEST',delimiter=',')
	X_train, X_test, y_train, y_test = [],[], [], []

	y_train = data_train[:,0]-1
	y_test = data_test[:,0]-1

	for index, x in enumerate(data_train):
		X_train.append(data_train[index][1:74])
	X_train = np.array( X_train)

	for index, x in enumerate(data_test):
		X_test.append(data_test[index][1:74])
	X_test = np.array(X_test)
	pd.DataFrame(X_train).to_csv("/Users/apple/Desktop/dev/projectlife/data/HAR_projectlife/train/X_train.txt", index = False, header=False, sep=",", encoding='utf-8')
	pd.DataFrame(y_train).to_csv("/Users/apple/Desktop/dev/projectlife/data/HAR_projectlife/train/y_train.txt", index = False, header=False, sep=",", encoding='utf-8')
	pd.DataFrame(X_test).to_csv("/Users/apple/Desktop/dev/projectlife/data/HAR_projectlife/test/X_test.txt", index = False, header=False, sep=",", encoding='utf-8')
	pd.DataFrame(y_test).to_csv("/Users/apple/Desktop/dev/projectlife/data/HAR_projectlife/test/y_test.txt", index = False, header=False, sep=",", encoding='utf-8')

if __name__ == '__main__':
	#prepare_cnn_patterns("candle")
	prepare_cnn_ticker("candle")
	#prepare_HAR_data()
	#prepare_UCR_data()
	#prepare_UCR_data_full()
