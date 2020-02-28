import csv
import json
import pdb
import pandas as pd
#from talib.abstract import *
#import talib
import datetime
from sklearn.model_selection import train_test_split

SYMBOLS_BTC = [

            "NEOBTC"

		]
SYMBOLS_USDT = [
		"BTCUSDT",
		"DASHUSDT",
		"ETHUSDT",
		"LTCUSDT",
		"XMRUSDT",
		"XRPUSDT"
	]
SYMBOLS_SINGLE =["POEBTC", "OMGBTC","ONEBTC","ONGBTC","ONTBTC","OSTBTC","PHBBTC","PIVXBTC","POABTC","HOTBTC","ICXBTC","KEYBTC","KNCBTC","TRXBTC","STORJBTC","MTHBTC","RLCBTC","RCNBTC","NANOBTC", "NEOBTC", "GASBTC", "DLTBTC","BCDBTC","OAXBTC", "ARNBTC","TNTBTC"]
target = "BTC"
full_data = True
timeframe ="5m"
seperator = ','
dateformat = '%Y-%m-%d %H:%M'
provider = "freqtrade"

for pair in SYMBOLS_BTC:
	if pair !=  "BTCBBTC":
		json_path = "/Users/apple/Desktop/dev/projectlife/utils/freqtrade/user_data/data/binance/"+pair.split(target)[0]+"_"+target+"-"+timeframe+".json"
	else:
		json_path = "/Users/apple/Desktop/dev/projectlife/utils/freqtrade/user_data/data/binance/BTCB_BTC-"+timeframe+".json"
	train_path = "/Users/apple/Desktop/dev/projectlife/data/train/Binance_"+pair+"-"+timeframe+".csv"
	validate_path = "/Users/apple/Desktop/dev/projectlife/data/validate/Binance_"+pair+"-"+timeframe+".csv"
	full_path = "/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+pair+"-"+timeframe+".csv"
	print(json_path)
	df = pd.read_json(json_path)
	df.columns=["date","open","high","low","close","volume"]
	df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime(dateformat)
	#pdb.set_trace()
	df.drop_duplicates(subset=None, inplace=True)
	# df["EMA5"] = EMA(df, timeperiod=5, price='close')
	# df["EMA20"] = EMA(df, timeperiod=20, price='close')
	# df["EMA40"] = EMA(df, timeperiod=40, price='close')
	# df["RSI"] = RSI(df, price='close')
	# df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
	#new = df[["close"]].copy()

	new = df[["date","open", "high","low","close","volume"]].copy()
	#new = df[["time","close","volume","RSI","MACD"]].copy()
	if full_data == True:
		train, test = train_test_split(new, test_size=0.00001,shuffle=False)
		#pdb.set_trace()
		#train.to_dense().to_csv(full_path, index = False, sep=seperator, encoding='utf-8')
		train.to_csv(full_path, index = False, sep=seperator, encoding='utf-8')
	else:
		train, test = train_test_split(new, test_size=0.2,shuffle=False)
		train.to_dense().to_csv(train_path, index = False, sep=seperator, encoding='utf-8')
		test.to_dense().to_csv(validate_path, index = False, sep=seperator, encoding='utf-8')

