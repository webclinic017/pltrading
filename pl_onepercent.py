from dateutil import parser
from configparser import ConfigParser
import pandas as pd
import pdb
#import ccxt
import time
import numpy as np

interval = "1h"
config = ConfigParser()
config.read("config.ini")
#exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})
window_size = 10
transaction_fee = 0.00125
initial_balance = 100
BUY, SELL, HOLD = 0, 1, 2
results = {}
profit_perc = 1.11
stoploss_perc = 0.98
dateformat = '%Y-%m-%d-%H-%M'
datatype ="local"

def backtest():
	SYMBOLS = ["ETC/BTC"]
	for symbol in SYMBOLS:
		if datatype == "remote":
			data_base = exchange.fetch_ohlcv(symbol, interval,limit=1000)
			df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
			#date = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
			symbol =  symbol.split("/")[0] + symbol.split("/")[1]
		elif datatype =="local":
			symbol =  symbol.split("/")[0] + symbol.split("/")[1]
			data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
			df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		# df['change1'] = df.close.pct_change(periods=1)*100
		# df['change2'] = df.close.pct_change(periods=2)*100
		# df['change3'] = df.close.pct_change(periods=3)*100
		# df['change4'] = df.close.pct_change(periods=4)*100
		# df['change5'] = df.close.pct_change(periods=5)*100
		# df['change6'] = df.close.pct_change(periods=6)*100
		# df['change7'] = df.close.pct_change(periods=7)*100
		# df['change8'] = df.close.pct_change(periods=8)*100
		# df['change9'] = df.close.pct_change(periods=9)*100
		# df['change10'] = df.close.pct_change(periods=10)*100
		# df['change11'] = df.close.pct_change(periods=11)*100
		# df['change12'] = df.close.pct_change(periods=12)*100
		# df['change13'] = df.close.pct_change(periods=13)*100
		# df['change14'] = df.close.pct_change(periods=14)*100
		df['change1'] = df.close.pct_change(periods=1)*100
		df['change3'] = df.close.pct_change(periods=3)*100
		df['change6'] = df.close.pct_change(periods=6)*100
		df['change12'] = df.close.pct_change(periods=12)*100
		df['change24'] = df.close.pct_change(periods=24)*100
		df['change48'] = df.close.pct_change(periods=48)*100
		df['change72'] = df.close.pct_change(periods=72)*100
		df['change96'] = df.close.pct_change(periods=96)*100
		df['change120'] = df.close.pct_change(periods=120)*100
		df['change144'] = df.close.pct_change(periods=144)*100
		df['change168'] = df.close.pct_change(periods=168)*100
		df = df.drop(["volume", "low", "high","open" ], axis=1)
		df.tail(300).to_csv("test.csv")
		#pdb.set_trace()
		#evaluate_trades(df,symbol,SYMBOLS)



def evaluate_trades(df,symbol):
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
		current_tick += 1
		if i > window_size:
			last_row =  df.loc[i]
			prev1 =  df.loc[i-1]
			prev2 =  df.loc[i-2]
			prev3 =  df.loc[i-3]

			#### BUY CONDITIONS #####
			cond1 = prev3.change > prev2.change and  prev3.change > prev1.change and  prev3.change > last_row.change
			cond2 = last_row.change > prev1.change and  last_row.change > prev2.change
			cond3 = prev3.change > 5 and last_row.change < 0 and last_row.change > -1 and prev1.change < 0 and prev2.change < 0
			buy_cond = (cond1 == True and cond2 == True and cond3 == True)

			#### SELL CONDITIONS #####
			sell_cond = (last_row.change > prev1.change)

			if buy_cond and buy_mode:
				buy_index = i
				action = BUY
				entry_price =  current_price
				entry_tick = current_tick
				quantity = balance / entry_price
				buy_mode = False
				print("##### TRADE " +  str(trade_count) + " #####")
				print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(last_row.date))
				fragment =df.iloc[i-window_size:i+2,:]
				print(fragment)
			elif not buy_mode and sell_cond:
				action = SELL
				sell_type = "profit"
				if df.iloc[buy_index+1].change > 8:
					exit_price = df.iloc[buy_index+1].close
				else:
					exit_price = current_price

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
if __name__ == '__main__':
	backtest()