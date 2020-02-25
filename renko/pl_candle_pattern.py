#ADXBTC - 2019-06-27. 30m 30ma-15ma up, no spike. olur.
##ADXBTC - 2019-06-24. 30m 30ma-15ma up, spike. olmaz.
#Go 2019-06-27 07:30 HABERCİ1!!!!
#ADX 60M http://prntscr.com/o7rk2k 2019-05-14 20:00
#http://prntscr.com/o7rp71

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
image_path = "/Users/apple/Desktop/dev/projectlife/data/images/candlestick/"
pattern_dir = "/Users/apple/Desktop/dev/projectlife/utils/classification/strict_patterns.json"

interval = "5m"
config = ConfigParser()
config.read("config.ini")
exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})
pd.set_option("display.precision", 8)
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 30)
dateformat = '%Y-%m-%d %H:%M'
dateformat_save = '%Y-%m-%d-%H-%M'
datatype = "local"

def fetch_symbol(symbol=None):
	if datatype =="local":
		data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
		df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		df['date'] = pd.to_datetime(df['date'])
	else:
		data_base = exchange.fetch_ohlcv(symbol, interval,limit=100)
		df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close','volume'])
		df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
	#df.set_index('date')
	return df

def backtest():
	if datatype =="local":
		save_predfined_patterns()
		#save_all_cross()
	else:
		exchange.load_markets()
		SYMBOLS = exchange.symbols
		for symbol in SYMBOLS:
			active = exchange.markets[symbol]['active']
			if active == True and ("/USDT" in symbol):
				print(symbol)
				#df = fetch_symbol(symbol)
				#df = df.head(80)
				#ohlc = create_ohlc_fragment(df)
				#savefig(ohlc,symbol)
				#is_hammer = talib.CDLHAMMER(df.open, df.high, df.low, df.close)
				#if is_hammer.values[-1] != 0:
				#pdb.set_trace()

def create_ohlc_fragment(df):
	df["date"] = df.index.values
	ohlc= df[['date', 'open', 'high', 'low','close']].copy()
	return ohlc.values

def save_predfined_patterns():
	data_start_date = pd.to_datetime("2017-08-01 00:00")
	with open(pattern_dir) as json_file:
		data = json.load(json_file)
		df_array = []
		for elem in data:
			try:
				symbol = elem['symbol']
				pattern = elem['patterns'][interval]
			except:
				pdb.set_trace()
			if pattern != []:
				df = fetch_symbol(symbol)
				patterns = []
				for p in pattern:
					end = pd.to_datetime(p["end"])
					if end > data_start_date:
						patterns.append(p["end"])
				for i, row in df.iterrows():
					row_date = row.date.strftime(dateformat)
					if row_date  in patterns:
						for p in pattern:
							end = pd.to_datetime(p["end"])
							if row_date == end.strftime(dateformat):
								fragment = df.iloc[i-72:i+1,:]
								ohlc = create_ohlc_fragment(fragment)
								savefig(ohlc,symbol)

def savefig(df, symbol):
	plt.cla()
	fig,ax = plt.subplots(figsize = (16,16))
	candlestick_ohlc(ax, df, width=0.6, colorup='green', colordown='red' )
	#symbol =  symbol.split("/")[0] + symbol.split("/")[1]
	fn = image_path+symbol+".png"
	plt.savefig(fn, dpi=100)

if __name__ == '__main__':
	backtest()