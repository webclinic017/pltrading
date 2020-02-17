#http://marketsurvival.net/trading-strategies/renko-trading-strategy-forex-stocks/
from sklearn.model_selection import GridSearchCV
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
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import glob
import os

class Renko:
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
		self.atr_coeff_params = [6]#[0.25,0.5,0.75,1,1.25,1.50,1.75,2,2.25,2.5,2.75,3,3.25,3.5,4] #"BTCUSDT"->3, Trx-> 0.5, "CNDBTC"->3.5,  linkbtc->7

		if (self.datatype =="local"):
			self.symbols =  ["BCDBTC"] #
			#self.symbols = ["ADABTC","ADXBTC","AEBTC","AGIBTC","AIONBTC","ALGOBTC","AMBBTC","APPCBTC","ARDRBTC","ARKBTC","ARNBTC","ASTBTC","ATOMBTC","BCDBTC","BCHBTC","BCPTBTC","BLZBTC","BNBBTC","BNTBTC","BQXBTC","BRDBTC", "BTGBTC","BTSBTC","BTTBTC","CDTBTC","CELRBTC","CMTBTC","CNDBTC","CVCBTC","DASHBTC","DATABTC","DCRBTC","DENTBTC","DGDBTC","DLTBTC","DNTBTC","DOCKBTC","EDOBTC","ELFBTC","ENGBTC","ENJBTC","EOSBTC","ERDBTC","ETCBTC","ETHBTC","EVXBTC","FETBTC","FTMBTC","FUELBTC","FUNBTC","GASBTC","GNTBTC","GOBTC","GRSBTC","GTOBTC","GVTBTC","GXSBTC","HCBTC","HOTBTC","ICXBTC","INSBTC","IOSTBTC","IOTABTC","IOTXBTC","KEYBTC","KMDBTC","KNCBTC","LENDBTC","LINKBTC","LOOMBTC","LRCBTC","LSKBTC","LTCBTC","LUNBTC","MANABTC","MATICBTC","MCOBTC","MDABTC","MFTBTC","MITHBTC","MTHBTC","MTLBTC","NANOBTC","NASBTC","NAVBTC","NCASHBTC","NEBLBTC","NEOBTC","NPXSBTC","NULSBTC","NXSBTC","OAXBTC","OMGBTC","ONEBTC","ONGBTC","ONTBTC","OSTBTC","PHBBTC","PIVXBTC","POABTC","POEBTC","POLYBTC","POWRBTC","PPTBTC","QKCBTC","QLCBTC","QSPBTC","QTUMBTC","RCNBTC","RDNBTC","RENBTC","REPBTC","REQBTC","RLCBTC","RVNBTC","SCBTC","SKYBTC","SNGLSBTC","SNMBTC","SNTBTC","STEEMBTC","STORJBTC","STORMBTC","STRATBTC","SYSBTC","TFUELBTC","THETABTC","TNBBTC","TNTBTC","TRXBTC","VETBTC","VIABTC","VIBBTC","VIBEBTC","WABIBTC","WANBTC","WAVESBTC","WPRBTC","WTCBTC","XEMBTC","XLMBTC","XMRBTC","XRPBTC","XVGBTC","XZCBTC","YOYOWBTC","ZECBTC","ZENBTC","ZILBTC","ZRXBTC"]
		    #self.symbols = ["ADAUSDT", "ALGOUSDT", "ATOMUSDT", "BATUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "BTTUSDT", "CELRUSDT", "DASHUSDT", "DOGEUSDT", "ENJUSDT", "EOSUSDT", "ERDUSDT", "ETCUSDT", "ETHUSDT", "FETUSDT", "FTMUSDT", "GTOUSDT", "HOTUSDT", "ICXUSDT", "IOSTUSDT", "IOTAUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "MITHUSDT", "NANOUSDT", "NEOUSDT", "NULSUSDT", "OMGUSDT", "ONEUSDT", "ONGUSDT", "ONTUSDT", "PAXUSDT", "QTUMUSDT", "TFUELUSDT", "THETAUSDT", "TRXUSDT", "TUSDUSDT", "USDCUSDT", "USDSUSDT", "USDSBUSDT", "VETUSDT", "WAVESUSDT", "XLMUSDT", "XMRUSDT", "XRPUSDT", "ZECUSDT", "ZILUSDT", "ZRXUSDT"]
		else:
			exchange.load_markets()
			self.symbols = exchange.symbols

	def start_backtest(self):
		for symbol in self.symbols:
			df = self.fetch_symbol(symbol)
			for atr_coeff in self.atr_coeff_params:
				print(symbol + "-ATR COEFFICENT: "+str(atr_coeff))
				renko_df = self.populate_renko(df,atr_coeff)
				#self.evaluate_trades(renko_df,symbol,save_chunks=False,show_trades=True)
				#self.print_df(renko_df)
				self.plot_renko(renko_df,symbol,"show")

	def populate_renko(self, df,atr_coeff):
		df['atr'] = talib.ATR(df['high'],df['low'],df['close'],14)
		brick_size = np.mean(df['atr']) * atr_coeff
		cdf = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'atr'],  data=[])
		cdf.loc[0] = df.loc[0]
		close = df.loc[0]['close']
		cdf.loc[0, 1:] = [close - brick_size, close, close - brick_size, close, brick_size]
		cdf['uptrend'] = True
		columns = ['date', 'open', 'high', 'low', 'close', 'atr', 'uptrend']
		for index, row in df.iterrows():
			if not np.isnan(row['atr']): brick_size = row['atr']
			close = row['close']
			date = row['date']
			row_p1 = cdf.iloc[-1]
			uptrend = row_p1['uptrend']
			close_p1 = row_p1['close']
			bricks = int((close - close_p1) / brick_size)
			data = []
			if uptrend and bricks >= 1:
				for i in range(bricks):
					r = [date, close_p1, close_p1 + brick_size, close_p1, close_p1 + brick_size,   brick_size, uptrend]
					data.append(r)
					close_p1 += brick_size
			elif uptrend and bricks <= -2:
				uptrend = not uptrend
				bricks += 1
				close_p1 -= brick_size
				for i in range(abs(bricks)):
					r = [date, close_p1, close_p1, close_p1 - brick_size, close_p1 - brick_size,   brick_size, uptrend]
					data.append(r)
					close_p1 -= brick_size
			elif not uptrend and bricks <= -1:
				for i in range(abs(bricks)):
					r = [date, close_p1, close_p1, close_p1 - brick_size, close_p1 - brick_size,  brick_size, uptrend]
					data.append(r)
					close_p1 -= brick_size
			elif not uptrend and bricks >= 2:
				uptrend = not uptrend
				bricks -= 1
				close_p1 += brick_size
				for i in range(abs(bricks)):
					r = [date, close_p1, close_p1 + brick_size, close_p1, close_p1 + brick_size,  brick_size, uptrend]
					data.append(r)
					close_p1 += brick_size
			else:
				continue
			sdf = pd.DataFrame(data=data, columns=columns)
			cdf = pd.concat([cdf, sdf])
		renko_df = cdf.reset_index()
		renko_df = renko_df.drop(['high',"low"], axis=1)
		renko_df['sma5'] = talib.SMA(renko_df['close'],5)
		renko_df['sma10'] = talib.SMA(renko_df['close'],10)
		renko_df['sma15'] = talib.SMA(renko_df['close'],15)
		renko_df['sma20'] = talib.SMA(renko_df['close'],20)
		return renko_df

	def print_df(self,df):
		with pd.option_context('display.max_rows', None):
			print(df)

	def evaluate_trades(self,renko_df,symbol,save_chunks=True,show_trades=True):
		self.balance = self.initial_balance
		trade_count = 0
		wincount = 0
		losscount = 0
		profit = 0
		action = 0
		trade_history = []
		current_tick = 0
		entry_tick = 0
		buy_mode = True
		entry_price = 0
		buy_index = 0
		sell_index = 0
		self.results[symbol] = {'balance':np.array([self.balance]), "trade_history":trade_history, "trade_count":trade_count }
		for i, row in renko_df.iterrows():
			current_tick += 1
			if i > 5 and i < len(renko_df) -1:
				last = renko_df.iloc[i]
				current_price = last['close']
				prev1,prev2,prev3,prev4,prev5 = renko_df.iloc[i-1],renko_df.iloc[i-2],renko_df.iloc[i-3],renko_df.iloc[i-4],renko_df.iloc[i-5]
				pattern_up =(last.uptrend == True and prev1.uptrend == False)# and prev2.uptrend == True and prev3.uptrend == True and prev4.uptrend == False)# and prev4.uptrend == True and prev5.uptrend == False) ###prev2den oteye bozuluyor
				brick_sma_up = last.close > last.sma20 and prev3.close > prev3.sma20 #and prev4.close > pre4.sma20 - prev5.close > pre5.sma20### olmuyor
				brick_sma_down = last.close < last.sma20 and prev3.close < prev3.sma20
				cross_down = last.sma5 < last.sma10 and prev5.sma10 < prev5.sma5
				#cross_up = last.sma5 > last.sma10 and prev5.sma10 > prev5.sma5  ### olmuyor
				#cross_up_big =  last.sma200 > prev5.sma200 #last.sma100 > prev10.sma100 and  ### olmuyor
				#stable_up_sma5 = last.sma5 > prev1.sma5 and prev1.sma5 > prev2.sma5 #and prev2.sma5 > prev3.sma5 ### olmuyor
				#stable_up_sma10 = last.sma10 > prev1.sma10 and prev1.sma10 > prev2.sma10 and prev2.sma10 > prev3.sma10 ### olmuyor
				#prev_cross_up = prev1.sma5 > prev1.sma10 and prev2.sma5 > prev2.sma10 and prev3.sma5 > prev3.sma10 ### olmuyor
				pattern_down =(last.uptrend == False and prev1.uptrend == True)# and prev2.uptrend == False and prev3.uptrend == True) ### olmuyor
				stable_up_all = last.sma5 > prev5.sma5 #and last.sma10 > prev5.sma10 and last.sma20 > prev5.sma20


				#### BUY CONDITIONS #####
				buy_cond = pattern_up and brick_sma_up #  and  stable_up_all #
				#### SELL CONDITIONS #####
				sell_cond = cross_down #and pattern_down# and brick_sma_down#

				if buy_cond and buy_mode and sell_index < i:
					buy_index = i
					action = 1
					entry_price =  current_price
					for x in range(10):
						next_renko_index = renko_df.iloc[i+x+1].values[0]
						if next_renko_index == 0:
							entry_price = renko_df.iloc[i+x].close
							break
						else:
							action = 0
					if action == 0:
						continue
					entry_tick = current_tick
					quantity = self.balance / entry_price
					buy_mode = False
					if show_trades:
						print("##### TRADE " +  str(trade_count) + " #####")
						print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(last.date))
					if save_chunks and i > 20:
						fragment = renko_df.iloc[i-15:i+5,:]
						fragment = fragment.reset_index()
						self.plot_renko(fragment,symbol,"save",buy_index=buy_index)
				elif not buy_mode and sell_cond:
					action = 2
					if i+3 < len(renko_df):
						for x in range(10000):
							renko_index = renko_df.iloc[i+x].values[0]
							sell_index = i+x
							next_renko_index = renko_df.iloc[i+x+1].values[0]
							if next_renko_index == 0:
								exit_price = renko_df.iloc[i+x].close
								break
					profit = ((exit_price - entry_price)/entry_price + 1)*(1-self.transaction_fee)**2 - 1
					self.balance = self.balance * (1.0 + profit)
					entry_price = 0
					trade_count += 1
					if profit > 0:
						wincount += 1
					else:
						losscount += 1
					buy_mode = True
					if show_trades:
						print("buy at: "+str(buy_index))
						print("df sell at: "+str(sell_index))
						print("renko sell at: "+str(renko_index))
						print("SELL: " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(last.date) )
						print("PROFIT: " + str(profit*100))
						print("BALANCE: " + str(self.balance))
						print("==================================")
					if save_chunks:
						self.update_renko_plot(symbol,buy_index, sell_index, profit)
				else:
					action = 0

				trade_history.append((action, current_tick, current_price, self.balance, profit))


			if (current_tick > len(renko_df)-1):
				self.results[symbol] = {'balance':np.array([self.balance]), "trade_history":trade_history, "trade_count":trade_count }
				print("**********************************")
				print("TOTAL BALANCE FOR "+symbol +": "+ str(self.balance))
				print("**********************************")
				print("WIN COUNT: "+str(wincount))
				print("LOSS COUNT: "+str(losscount))

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
		#df.set_index('date')
		return df

	def update_renko_plot(self,symbol,buy_index, sell_index, profit):
		files =  glob.glob("/Users/apple/Desktop/dev/projectlife/data/images/renko/*.png")
		for file in files:
			if (file.split("-buy-")[-1].split(".png")[0]) == str(buy_index):
				if profit > 0:
					pro = "WIN:"+ str(profit *100)
				else:
					pro = "LOSS:"+ str(profit *100)
				x = file.split(".png")[0] + "-sell-"+str(sell_index) + "-"+pro+".png"
				os.rename(file, x)

	def plot_renko(self,df,symbol,type,buy_index=None):
		num_bars = len(df)
		df = df.tail(num_bars)
		renkos = zip(df['open'],df['close'],df['date'] )
		price_move = abs(df.iloc[1]['open'] - df.iloc[1]['close'])
		fig = plt.figure(1)
		fig.clf()
		axes = fig.gca()
		index = 1
		for open_price, close_price, date in renkos:
			if (open_price < close_price):
				renko = Rectangle((index,open_price), 1, close_price-open_price, edgecolor='darkblue', facecolor='blue', alpha=0.5)
				axes.add_patch(renko)
			else:
				renko = Rectangle((index,open_price), 1, close_price-open_price, edgecolor='darkred', facecolor='red', alpha=0.5)
				axes.add_patch(renko)
			index = index + 1
		#plt.plot([0, num_bars],[min(min(df['open']),min(df['close'])), max(max(df['open']),max(df['close']))] )
		#plt.plot(df.index,df.sma10,color="purple", marker=".")
		#plt.plot(df.index,df.sma5, color="orange", marker=".")
		plt.plot(df.index,df.sma20, color="red" marker=".")
		fig.suptitle(symbol)
		#plt.xlim([0, num_bars])
		#plt.ylim([min(min(df['open']),min(df['close'])), max(max(df['open']),max(df['close']))])
		#plt.xlabel('Bar Number')
		#plt.ylabel('Price')
		plt.grid(True)
		#line = Line2D(df.index, df['sma10'])
		#axes.add_line(line)
		if type == "show":
			self.print_df(df)
			plt.show()
		else:
			path = "/Users/apple/Desktop/dev/projectlife/data/images/renko/"+symbol+"-buy-"+str(buy_index)+".png"
			fig.savefig(path)

def main():
	backtest = Renko()
	backtest.start_backtest()

if __name__ == "__main__":
	main()

