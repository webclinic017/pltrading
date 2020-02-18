import csv
import json
import pdb
import pandas as pd
import os

import datetime
from sklearn.model_selection import train_test_split


symbols = os.listdir( "/home/ubuntu/datacollecttemp/")

target = "BTC"
full_data = True
timeframe ="1m"
seperator = ','
dateformat = '%Y-%m-%d %H:%M'
provider = "freqtrade"

for symbol in symbols:
	symbol =  symbol.split(".csv")[0]
	path = "/home/ubuntu/datacollecttemp/"+symbol+".csv"
	df = pd.read_csv(path)
	df.columns=['symbol','date','price_change','price_change_percent','last_price','best_bid_price',
				'best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
	df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime(dateformat)
	df.to_csv(path, index = False, sep=seperator, encoding='utf-8')
	print(symbol)