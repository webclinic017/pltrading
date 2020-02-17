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
import pandas as pd
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
import pdb
import matplotlib.pyplot as plt
import numpy as np
import ccxt
import time
from pyod.models.knn import KNN
pd.set_option("display.precision", 8)
pd.set_option('display.max_rows', 1500)

datatype ="remote"
transaction_fee = 0.00125
initial_balance = 100
BUY, SELL, HOLD = 0, 1, 2
results = {}
interval = "1h"
config = ConfigParser()
config.read("config.ini")
exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})
window_size = 64
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
		#save_symbols(SYMBOLS)
		for symbol in SYMBOLS:
			symbol = symbol.split("/")[0] + symbol.split("/")[1]
			df = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/allin/Binance_"+symbol+"-"+interval+".csv")
			df = df.tail(144)
			#pdb.set_trace()
			evaluate_symbol(df,symbol,SYMBOLS)
	else:
		SYMBOLS = ["TNTBTC","ATOMBTC", "BCDBTC","BTGBTC","SKYBTC", "CELRBTC","TRXBTC"] #,"DATABTC", "QTUMBTC"]#,"HCBTC","RDNBTC","LTCBTC", "REPBTC", "QKCBTC","IOTABTC","KMDBTC","ZECBTC", "DLTBTC", "ENJBTC","GOBTC","LINKBTC","EDOBTC","GRSBTC","APPCBTC","ATOMBTC","ARKBTC","NULSBTC","SNGLSBTC","TNTBTC","WAVESBTC"]
		for symbol in SYMBOLS:
			if datatype == "remote":
				data_base = exchange.fetch_ohlcv(symbol.split("BTC")[0] + "/BTC", interval,limit=1000)
				df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
				df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
			elif datatype =="local":
				data_base = read_csv("/Users/apple/Desktop/dev/projectlife/data/full/Binance_"+symbol+"-"+interval+".csv")
				df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
			evaluate_symbol(df,symbol,SYMBOLS)

def evaluate_symbol(df,symbol,SYMBOLS):
	df = df.drop(['open'], axis=1)
	df.set_index('date')
	df['change'] = df.close.pct_change(periods=1)*100
	#df['low_change'] = df.low.pct_change(periods=1)*100
	#df['high_change'] = df.high.pct_change(periods=1)*100
	df["rmean10"] = df.change.rolling(10).mean()
	df["rmean30"] = df.change.rolling(30).mean()
	df["rmean50"] = df.change.rolling(50).mean()
	df = df.fillna(0)
	#plot_fragment(df,symbol)
	evaluate_trades(df,symbol,SYMBOLS)
	df = detect_anomaly(df)
	#fragment = df.iloc[i-window_size:i+8,:]

def allin(df,i,buy_index):
	buy_cond, sell_cond, stopl_cond = False, False, False
	if i-3 >= df.iloc[0].name:
		last_row =  df.loc[i]
		prev3 =  df.loc[i-3]
		prev2 =  df.loc[i-2]
		prev1 =  df.loc[i-1]
		buy_rmean_cond =  last_row.rmean10 > prev1.rmean10
		buy_change_cond = last_row.change > 0 and prev1.change > 0 #and last_row.change + prev1.change > 2
		sell_change_cond = last_row.change > prev1.change and prev2.change > prev1.change
		buy_cond =  buy_change_cond and buy_rmean_cond
		sell_cond = sell_change_cond
		stopl_cond = False
	return buy_cond, sell_cond, stopl_cond


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
		high_price = last_row['high']
		current_tick += 1
		fragment = df.iloc[i-window_size:i+8,:]
		buy_cond, sell_cond, stopl_cond =  spikedetect(df,i,buy_index)
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
			pdb.set_trace()
			action = SELL
			if stopl_cond:
				sell_type = "stoploss"
				exit_price =  current_price
			else:
				sell_type = "profit"
				exit_price = current_price# (entry_price * profit_perc) #last_row['high']
			profit = ((exit_price - entry_price)/entry_price + 1)*(1-transaction_fee)**2 - 1
			balance = balance * (1.0 + profit)
			entry_price = 0
			trade_count += 1
			if profit <= 0:
				loss_count += 1
			else:
				win_count += 1
			buy_mode = True
			print(fragment)
			print(sell_type)
			print("buy at: "+str(buy_index))
			print("sell at: "+str(i))
			print("SELL FOR " + sell_type.upper() +" : " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(last_row.date) )
			print("PROFIT: " + str(profit))
			print("BALANCE: " + str(balance))
			print("==================================")
		else:
			action = HOLD

		trade_history.append((action, current_tick, current_price, balance, profit))

		if (current_tick > len(df)-1):
			results[symbol] = {'balance':np.array([balance]), "trade_history":trade_history, "trade_count":trade_count }
			print("**********************************")
			print("TOTAL BALANCE FOR "+symbol +": "+ str(balance))
			print("**********************************")
			#plot_buy_sell(trade_history)

	if (symbol == SYMBOLS[-1]):
		print("#########FINAL BALANCES#####################")
		final_balance = 0
		for symbol in SYMBOLS:
			balance = results[symbol]['balance'][0]
			print(symbol + ": "+str(balance))
			final_balance +=balance
		print("============================================")
		print("Trade count: " + str(trade_count))
		print("Win count: " + str(win_count))
		print("Loss count: " + str(loss_count))
		print("FINAL BALANCE: " + str(final_balance))


def spikedetect(df,i,buy_index):
	buy_cond, sell_cond, stopl_cond = False, False, False
	if i-3 >= df.iloc[0].name:
		last_row =  df.loc[i]
		prev3 =  df.loc[i-3]
		prev2 =  df.loc[i-2]
		prev1 =  df.loc[i-1]
		buy_rmean_cond =  last_row.rmean10 > prev1.rmean10 and last_row.rmean10 > last_row.rmean30 and last_row.rmean10 > last_row.rmean50 and prev1.rmean10 < prev1.rmean30 and prev1.rmean10 < prev1.rmean50
		buy_change_cond = last_row.change + prev1.change > 2
		sell_change_cond =  last_row.rmean10 < prev1.rmean10 #last_row.change  prev1.change  and prev2.change > prev1.change and prev2.change > prev3.change and last_row.change > 0 and prev1.change > 0  and prev2.change > 0
		buy_cond =  buy_change_cond and buy_rmean_cond
		sell_cond = sell_change_cond
	return buy_cond, sell_cond, stopl_cond

def plot_fragment(df,symbol):
	fragment = df#.tail(200)
	plt.clf()
	plt.title(symbol + " " +  str(fragment.iloc[-1].date))
	plt.plot(fragment.change, marker="o", label="change")
	plt.plot(fragment.rmean10, marker="o", label="rmean10")
	plt.plot(fragment.rmean30, marker="o", label="rmean30")
	plt.plot(fragment.rmean50, marker="o", label="rmean50")
	#plt.plot(fragment.high_change, marker="o", label="high change")
	#plt.plot(fragment.low_change, marker="o", label="low change")
	plt.legend()
	print(fragment)
	#pdb.set_trace()
	#plt.savefig("./results/plots/"+symbol+"-"+interval+".png")
	plt.show()

def plot_buy_sell(trade_history):
	closes = [data[2] for data in trade_history]
	closes_index = [data[1] for data in trade_history]
	buy_tick = np.array([data[1] for data in trade_history if data[0] == 0])
	buy_price = np.array([data[2] for data in trade_history if data[0] == 0])
	sell_tick = np.array([data[1] for data in trade_history if data[0] == 1])
	sell_price = np.array([data[2] for data in trade_history if data[0] == 1])
	plt.plot(closes_index, closes)
	plt.scatter(buy_tick, buy_price, c='g', marker="^", s=20)
	plt.scatter(sell_tick, sell_price , c='r', marker="v", s=20)
	#plt.savefig(base_path+"/results/plots/"+algo +"-"+pair+"-"+interval+"-"+str(balance)+".png" )
	plt.show(block=True)


def detect_anomaly(df):
	clf = KNN()
	x_values = df.change.values.reshape(df.index.values.shape[0],1)
	y_values = df.change.values.reshape(df.change.values.shape[0],1)
	clf.fit(y_values)
	clf.predict(y_values)
	df["out_label"] = clf.predict(y_values)  #fit_predict_score
	df["out_score"] = clf.decision_function(y_values)
	return df


def p_change(first, second):
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

def down_revert(df):
	prev3 =  df.loc[i-3]
	prev2 =  df.loc[i-2]
	prev1 =  df.loc[i-1]
	last20 = df.iloc[i-20:i,:]
	min = last20.change.min()
	max = last20.change.max()
	mean = last20.change.mean()
	max_down_mean = abs(max - min)
	low_mean =  (min - mean) *2
	potential_spike_down = last_row.change - prev1.change
	# if i == 935:
	# 	pdb.set_trace()
	mean_up = prev1.rmean10 < -1 and last_row.rmean10 > prev1.rmean10 and last_row.rmean10 < last_row.rmean30 and last_row.rmean30 < last_row.rmean50
	mean_down = last_row.rmean10 > last_row.rmean50 and last_row.rmean50 > last_row.rmean30
	spike_down = potential_spike_down < low_mean and last_row.change < -4 and prev1.change < 1
	buy_cond = (buy_mode == True and mean_up)
	sell_cond = (buy_mode == False and mean_down)# and (df.iloc[entry_tick].high > (entry_price * profit_perc)))
	stopl_cond = False#(sell_cond == False and buy_mode == False)
	#plot_fragment(df,symbol)


if __name__ == '__main__':
	backtest()
#df.to_csv("./results/pct_change/ast-7day.csv")

# def calc_hour_to_now_change(df,symbol):
# 	df = df.drop(['open', 'high', 'low', 'volume'], axis=1)
# 	df.replace({0: np.nan}, inplace=True)
# 	for i in range(72):
# 		df[str(i)+'_price_change'] = df.close.pct_change(periods=i)
# 	df = df.drop(['date','close'], axis=1)
# 	values = df.iloc[-1].values
# 	print(last_n)
# 	plt.clf()
# 	plt.plot(values)
# 	plt.plot()
# 	pdb.set_trace()

# def calc_arima(df,symbol):
# 	df.set_index('date')
# 	df['change'] = df.close.pct_change(periods=1)*100
# 	model = ARIMA(last_n.change.values, order=(5,1,0))
# 	model_fit = model.fit()
# 	model_fit.plot_predict(start=1, end=33)
# 	plt.show()