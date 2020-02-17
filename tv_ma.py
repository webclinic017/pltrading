#https://www.digitalocean.com/community/tutorials/understanding-class-and-instance-variables-in-python-3
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
pd.set_option("display.precision", 8)
pd.set_option('display.max_rows',250)
pd.set_option('display.max_columns', 30)


class Backtest:
	config = ConfigParser()
	config.read("config.ini")
	exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})
	dateformat = '%Y-%m-%d %H:%M'
	dateformat_save = '%Y-%m-%d-%H-%M'

	def __init__(self):
		self.datatype = "local"
		self.interval = "15m"
		self.transaction_fee = 0.00125
		self.initial_balance = 100
		self.results = {}
		self.datatype = "local"
		self.pattern_dir = "/Users/apple/Desktop/dev/projectlife/utils/classification/ma_patterns.json"


		if self.datatype =="local":
			#self.symbols=["BCDBTC"]
			#self.symbols =["BTCUSDT","LINKUSDT","ETHUSDT","LTCUSDT"]
			self.symbols = ["NASBTC", "ARDRBTC", "GXSBTC", "BCDBTC", "HCBTC", "TNTBTC", "BTGBTC", "ATOMBTC", "WANBTC", "EOSBTC", "SKYBTC"]
			#self.symbols =  ["NASBTC", "ARDRBTC", "GXSBTC", "BCDBTC", "HCBTC", "TNTBTC", "BTGBTC", "ATOMBTC", "WANBTC", "EOSBTC", "SKYBTC", "RDNBTC", "LTCBTC", "REPBTC", "QKCBTC", "IOTABTC", "ZECBTC", "DLTBTC", "ENJBTC", "GOBTC", "LINKBTC", "EDOBTC", "GRSBTC", "APPCBTC", "ARKBTC", "NULSBTC", "SNGLSBTC", "WAVESBTC", "BATBTC", "QTUMBTC", "DCRBTC", "HOTBTC", "RVNBTC", "LSKBTC", "OMGBTC", "ZILBTC", "ZRXBTC", "BCNBTC", "PAXBTC", "BTSBTC", "NPXSBTC", "ICXBTC", "XVGBTC", "DENTBTC", "STEEMBTC", "THETABTC", "SNTBTC", "ELFBTC", "MCOBTC", "STRATBTC", "GNTBTC", "MATICBTC", "TFUELBTC", "PPTBTC", "LRCBTC", "LOOMBTC", "MANABTC", "AIONBTC", "POWRBTC", "ARNBTC", "WABIBTC", "XZCBTC", "MITHBTC", "POABTC", "MTLBTC", "CVCBTC", "AGIBTC", "POEBTC", "GASBTC", "KMDBTC", "ASTBTC"]
		else:
			exchange.load_markets()
			self.symbols = exchange.symbols

	def start_backtest(self):
		for symbol in self.symbols:
			df = self.fetch_symbol(symbol)
			#self.evaluate_trades(df,symbol)
			save_predfined_patterns()
			#save_all_cross(df,symbol,SYMBOLS)

	def evaluate_trades(self,df,symbol):
		self.balance = self.initial_balance
		trade_count = 0
		profit = 0
		action = 0
		trade_history = []
		current_tick = 0
		entry_tick = 0
		buy_mode = True
		entry_price = 0
		buy_index = 0
		for i, last in df.iterrows():
			current_price = last['close']
			current_tick += 1
			if i > 15:
				prev2 =  df.loc[i-3]
				prev5 =  df.loc[i-5]
				cross_up = last.ema100 > last.ema200 and prev5.ema200 > prev5.ema100
				cross_down = last.ema100 < last.ema200 and prev5.ema200 < prev5.ema100
				stable_up = last.ema100 > prev2.ema100 and prev2.ema100 > prev5.ema100 and last.ema200 > prev2.ema200 and prev2.ema200 > prev5.ema200  and prev2.ema100 > prev2.ema200
				#### BUY CONDITIONS #####
				buy_cond = cross_up and stable_up

				#### SELL CONDITIONS #####
				sell_cond = cross_down

				if buy_cond and buy_mode:
					buy_index = i
					action = 1
					entry_price =  current_price
					entry_tick = current_tick
					quantity = self.balance / entry_price
					buy_mode = False
					print("##### TRADE " +  str(trade_count) + " #####")
					print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(last.date))
				elif not buy_mode and sell_cond:
					action = 2
					exit_price = current_price
					profit = ((exit_price - entry_price)/entry_price + 1)*(1-self.transaction_fee)**2 - 1
					self.balance = self.balance * (1.0 + profit)
					entry_price = 0
					trade_count += 1
					buy_mode = True
					print("buy at: "+str(buy_index))
					print("sell at: "+str(i))
					print("SELL: " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(last.date) )
					print("PROFIT: " + str(profit*100))
					print("BALANCE: " + str(self.balance))
					print("==================================")
				else:
					action = 0

			trade_history.append((action, current_tick, current_price, self.balance, profit))

			if (current_tick > len(df)-1):
				self.results[symbol] = {'balance':np.array([self.balance]), "trade_history":trade_history, "trade_count":trade_count }
				print("**********************************")
				print("TOTAL BALANCE FOR "+symbol +": "+ str(self.balance))
				print("**********************************")

		ax =self.symbols[-1]
		if (symbol == ax):
			print("#########FINAL BALANCES#####################")
			final_balance = 0
			for symbol in self.symbols:
				xbalance = self.results[symbol]['balance'][0]
				print(symbol + ": "+str(xbalance))
				final_balance +=xbalance
			print("============================================")
			print("Trade count: " + str(trade_count))
			print("Investment: " + str(len(self.symbols)*self.initial_balance))
			print("FINAL BALANCE: " + str(final_balance))

	def fetch_symbol(self,symbol=None):
		if self.datatype =="local":
			data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+self.interval+".csv")
			df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
			df['date'] = pd.to_datetime(df['date'])
		else:
			data_base = exchange.fetch_ohlcv(symbol, interval,limit=960)
			df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
			df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
		df = df.drop(['high',"low","open"], axis=1)
		df['symbol'] = symbol
		df.set_index('date')
		df['sma100'] = talib.SMA(df['close'],100)
		df['sma200'] = talib.SMA(df['close'],200)
		df['ema50'] = talib.EMA(df['close'],50)
		df['ema100'] = talib.EMA(df['close'],100)
		df['ema200'] = talib.EMA(df['close'],200)
		df['ema300'] = talib.EMA(df['close'],300)
		df['ema600'] = talib.EMA(df['close'],600)
		df['smma100'] = self.calculate_smma(df,100)
		df['smma200'] = self.calculate_smma(df,200)
		return df

	def calculate_smma(self,df,nx):
		smma_array = []
		for i, last in df.iterrows():
			if i > nx-2:
				fragment=df.iloc[i-(nx-1):i,:]
				if i == nx-1:
					sum1= fragment.close.sum()
					smma1 = float(sum1 / nx)
					smma_array.append(smma1)
				else:
					sum1= fragment.close.sum()
					smma1 = float(sum1 / nx)
					smmax =(sum1-smma1+last.close) / nx
					smma_array.append(smmax)
			else:
				smma_array.append(None)
		return smma_array

	def save_predfined_patterns(self):
		data_start_date = pd.to_datetime("2017-08-01 00:00")
		with open(pattern_dir) as json_file:
			data = json.load(json_file)
			df_array = []
			for elem in data:
				symbol = elem['symbol']
				pattern = elem['patterns'][interval]
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
									fragment = df.iloc[i-p["window"]:i+1,:]
									plot_ma(fragment,"save", name="predefined"+str(p["window"])+p["type"], date=row_date, symbol=symbol)


	def save_all_cross(self,df,symbol,SYMBOLS):
		#plot_ma(df,"show")
		found_pattern_index = 0
		total_profit = 0
		for i, row in df.iterrows():
			if i > 100:
				last =  df.loc[i]
				prev5 =  df.loc[i-5]
				cross_up1 = last.ema50 > last.ema200 and prev5.ema200 > prev5.ema50
				if cross_up1 and i > found_pattern_index + 15:
					found_pattern_index = i
					backward_window, buy_forward_window, sell_forward_window = 4, 36, 2000
					fragment_buy = df.iloc[i-backward_window:i+buy_forward_window,:]
					buy_row = fragment_buy.iloc[-1]
					buy_date = buy_row.date.strftime(dateformat_save)
					buy_price =  buy_row.close
					fragment_sell = df.iloc[i+buy_forward_window:i+sell_forward_window,:]
					final_index = fragment_sell.tail(1).index
					fragment_sell = fragment_sell.reset_index(drop=True)
					sell_price = 0
					for ix, sell_row in fragment_sell.iterrows():
						if ix > 5:
							prev5sr =  fragment_sell.loc[ix-5]
							cross_down = sell_row.ema50 < sell_row.ema200 and prev5sr.ema200 < prev5sr.ema50
							if cross_down:
								sell_price = sell_row.close
								break

					if sell_price != 0:
						change = pct_change(buy_price, sell_price)
						total_profit = total_profit + change
						if sell_price > buy_price:
							plot_ma(fragment_buy,"save","PROFITxx+"+str(change),buy_date, symbol)
						else:
							plot_ma(fragment_buy,"save","LOSSxx"+str(change),buy_date, symbol)
					else:
						plot_ma(fragment_buy,"save","END",buy_date, symbol)
		print(symbol + ": "+str(total_profit))


	def show_pattern_cross(self,df):
		for i, row in df.iterrows():
			if i > 120:
				last =  df.loc[i]
				prev10 =  df.loc[i-10]
				cross_up1 = last.sma100 > last.sma200 and prev10.sma200 > prev10.sma100
				if cross_up1:
					backward_window, forward_window = 10, 110
					fragment = df.iloc[i-backward_window:i+forward_window,:]
					plot_ma(fragment,"save","1short",last.date, symbol)


	def plot_ma(self,df,type, name=None, date=None, symbol=None):
		plt.clf()
		x = df['date']
		y1 = df['smma100']
		y2 = df['smma200']
		y3 = df['close']
		if type == "save":
			plt.plot(x, y1, color='blue')
			plt.plot(x, y2, color='blue')
			plt.fill_between(x, y1, y2, facecolor='blue')
			path = "/Users/apple/Desktop/dev/projectlife/data/images/ma/"+symbol+"-"+str(date)+"-"+name+".png"
			plt.axis('off')
			plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
			frame1 = plt.gca()
			frame1.axes.xaxis.set_ticklabels([])
			frame1.axes.yaxis.set_ticklabels([])
			plt.savefig(path, dpi=50)
		else:
			plt.plot(x, y1, color='red')
			plt.plot(x, y2, color='blue')
			plt.plot(x, y3, color='orange')
			plt.show()




	def pct_change(self,first, second):
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

def main():
	backtest = Backtest()
	backtest.start_backtest()

if __name__ == "__main__":
	main()

