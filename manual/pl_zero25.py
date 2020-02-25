from dateutil import parser
from configparser import ConfigParser
import pandas as pd
import pdb

import time
import numpy as np

interval = "1h"
config = ConfigParser()
config.read("config.ini")
window_size = 10
transaction_fee = 0.00125
initial_balance = 100
BUY, SELL, HOLD = 0, 1, 2
results = {}
profit_perc = 1.11
stoploss_perc = 0.98
dateformat = '%Y-%m-%d-%H-%M'
datatype ="local"
pd.set_option('display.max_rows', 500)

def backtest():
	SYMBOLS = ["BTC/USDT"]
	for symbol in SYMBOLS:
		symbol =  symbol.split("/")[0] + symbol.split("/")[1]
		data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
		df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		df['change'] = df.close.pct_change(periods=1)*100
		df = df.drop(["volume", "low", "high","open" ], axis=1)
		evaluate_trades(df,symbol)



def evaluate_trades(df,symbol):
	balance = initial_balance
	trade_count = 0
	win_count = 0
	loss_count = 0
	profit = 0
	action = HOLD
	trade_history = []
	current_tick = 0
	buy_mode = True
	entry_price = 0
	buy_index = 0
	######DASH
	data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_DASHUSDT-1h.csv")
	df_dash = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
	df_dash['change'] = df_dash.close.pct_change(periods=1)*100
	df_dash = df_dash.drop(["volume", "low", "high","open" ], axis=1)
	symbol = 'DASHUSDT'
	######DASH
	for i, last_row in df.iterrows():
		current_tick += 1
		if i > window_size:
			last_row =  df.loc[i]
			if last_row.change > 1:
				for ix, last_row_dash in df_dash.iterrows():
					last_row_dash =  df_dash.loc[ix]
					current_dash_price = last_row_dash['close']
					buy_cond = (last_row_dash.date == last_row.date)
					sell_cond = last_row_dash.change > 1
					stoploss_cond = last_row_dash.change < -2
					if buy_cond and buy_mode:
						buy_index = ix
						action = BUY
						entry_price = current_dash_price
						quantity = balance / entry_price
						buy_mode = False
						print("##### TRADE " +  str(trade_count) + " #####")
						print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(last_row_dash.date))
						fragment =df_dash.iloc[ix:ix+20,:]
						print(fragment)
					elif not buy_mode and sell_cond:
						action = SELL
						sell_type = "profit"
						exit_price = current_dash_price
						profit = ((exit_price - entry_price)/entry_price + 1)*(1-transaction_fee)**2 - 1
						balance = balance * (1.0 + profit)
						entry_price = 0
						trade_count += 1
						if profit <= 0:
							loss_count += 1
						else:
							win_count += 1
						buy_mode = True
						print("buy at: "+str(buy_index))
						print("sell at: "+str(ix))
						print("SELL FOR " + sell_type.upper() +" : " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(last_row_dash.date) )
						print("PROFIT: " + str(profit))
						print("BALANCE: " + str(balance))
						print("==================================")
						break
					elif not buy_mode and stoploss_cond:
						action = SELL
						sell_type = "stoploss"
						exit_price = current_dash_price
						profit = ((exit_price - entry_price)/entry_price + 1)*(1-transaction_fee)**2 - 1
						balance = balance * (1.0 + profit)
						entry_price = 0
						trade_count += 1
						if profit <= 0:
							loss_count += 1
						else:
							win_count += 1
						buy_mode = True
						print("buy at: "+str(buy_index))
						print("sell at: "+str(ix))
						print("SELL FOR " + sell_type.upper() +" : " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(last_row_dash.date) )
						print("STOPLOSS: " + str(profit))
						print("BALANCE: " + str(balance))
						print("==================================")
						break
					else:
						action = HOLD

					trade_history.append((action, current_tick, current_dash_price, balance, profit))

					if (current_tick > len(df)-1):
						results[symbol] = {'balance':np.array([balance]), "trade_history":trade_history, "trade_count":trade_count }
						print("**********************************")
						print("TOTAL BALANCE FOR "+symbol +": "+ str(balance))
						print("**********************************")
if __name__ == '__main__':
	backtest()