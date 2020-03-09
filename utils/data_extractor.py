from math import sqrt
from dateutil import parser
from configparser import ConfigParser
from pandas import Series
import matplotlib.dates as mdates
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
import time
from pandas import datetime
import platform
import pandas as pd
import pdb
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
from datetime import datetime, timedelta
import os

pd.set_option("display.precision", 9)
pd.set_option('display.max_rows', 3000)
pd.options.mode.chained_assignment = None

base_path = "/home/canercak/Desktop/dev/pltrading"
if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
	base_path = "/Users/apple/Desktop/dev/pltrading"
path = base_path +"/data/ticker3/"

def backtest():
	SYMBOLS =["KNCBTC"]
	# dir = os.listdir(path)
	# for s in dir:
	# 	if ".py" not in s and ".DS_Store" not in s:
	#  		SYMBOLS.append(s.split(".csv")[0])

	for symbol in SYMBOLS:
		data_base = read_csv(path+symbol+".csv")
		df = DataFrame(data_base)
		df.columns = ['symbol','date','price_change','price_change_percent','last_price','best_bid_price','best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
		df['last_sma100'] = df.last_price.rolling(100).mean()
		df['last_sma200'] = df.last_price.rolling(200).mean()
		df['last_sma600'] = df.last_price.rolling(600).mean()
		df = df.reset_index()
		df = df.fillna(0)
		window_size = 2000
		arr = []
		for i, row in df.iterrows():
			if i > window_size:
				fragment1 = df.iloc[i-window_size:i,:]
				fragment1 = detect_anomaly(fragment1)
				fragment1 = fragment1.reset_index()
				arr.append(fragment1)
				if i % 1000 == 0:
					print(symbol+"-"+str(row['index']))
		with open("file.txt", "w") as output_file:
		    for line in arr:
		       output_file.write("".join([str(i) for i in line]))
		       output_file.write("\n")

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
	backtest()


