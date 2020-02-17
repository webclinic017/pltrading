from dateutil import parser
from configparser import ConfigParser
import pandas as pd
import pdb
import ccxt
import time
import numpy as np
import talib
import json
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

image_path = "/Users/apple/Desktop/dev/projectlife/data/images/alltimehigh/"
interval = "1d"
config = ConfigParser()
config.read("config.ini")
exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})
pd.set_option("display.precision", 8)
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 30)
dateformat = '%Y-%m-%d %H:%M'
dateformat_save = '%Y-%m-%d-%H-%M'
datatype = "remote"

def fetch_symbol(symbol=None):
	if datatype =="local":
		data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
		df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		df['date'] = pd.to_datetime(df['date'])
	else:
		data_base = exchange.fetch_ohlcv(symbol, interval,limit=1000)
		df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close','volume'])
		df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
	#df.set_index('date')
	return df

def backtest():
	if datatype =="local":
		save_predfined_patterns()
	else:
		exchange.load_markets()
		SYMBOLS = exchange.symbols
		for symbol in SYMBOLS:
			active = exchange.markets[symbol]['active']
			if active == True and ("/BTC" in symbol):
				print(symbol)
				df = fetch_symbol(symbol)
				#plot_ohlc(df, symbol)
				plot_line(df, symbol)

def plot_ohlc(df, symbol):
	df["date"] = df.index.values
	ohlc= df[['date', 'open', 'high', 'low','close']].copy()
	ohlc_values = ohlc.values
	plt.cla()
	fig,ax = plt.subplots(figsize = (16,16))
	candlestick_ohlc(ax, ohlc_values, width=0.6, colorup='green', colordown='red' )
	symbol =  symbol.split("/")[0] + symbol.split("/")[1]
	fn = image_path+symbol+".png"
	plt.savefig(fn, dpi=100)

def plot_line(df,symbol):
	plt.clf()
	df = df.reset_index()
	plt.figure()
	plt.plot(df.close,label="close")
	symbol =  symbol.split("/")[0] + symbol.split("/")[1]
	fn = image_path+symbol+".png"
	plt.savefig(fn, dpi=200)


if __name__ == '__main__':
	backtest()