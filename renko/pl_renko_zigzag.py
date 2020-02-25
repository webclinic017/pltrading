#BTCUSDT-15m-atr 3 - http://prntscr.com/od3j9z
#BCDBTC-15m-atr 6 - http://prntscr.com/od3ecw
#BCPTBTC-15m-atr 6 - http://prntscr.com/od3tvs
#APPCBTC-15m-atr 6 - http://prntscr.com/od3w0b
#ARNBRC-15m-1tr- 6 - http://prntscr.com/od3z59
#BTGBTC -15m-1tr- 6 -http://prntscr.com/od4555
#CNDBTC -15m-1tr- 3 - http://prntscr.com/od4d8j - daha cok pattern apıor ama riskli
#DENTBTC -15m-1tr- 1 - http://prntscr.com/od4d8j - 2018-07'DE başlıyor bundan 6 atr olmuyor 3 veya aşağısı ancak. 1 atr ancak alıyor.
#DNTBTC -15m-1tr- 6 -http://prntscr.com/od4u16 - zaman uzadıkca atr artıyor.
#ENJBTC-15m-1tr- 6 - -http://prntscr.com/od4voe
#GRSBTC-15m-1tr- atr 6 olursa 1 tane http://prntscr.com/od4xws / 3 olursa 4 tane http://prntscr.com/od4zly / 2 olursa bozuyor
#LENDBTC -15m-1tr- 4 -  http://prntscr.com/od5bqj - dikkat!!!! olmayan durumlar var işaretli
#NEBLBTC -15m-1tr- 6 -  http://prntscr.com/od5dpy
#ETHUSDT -15m-1tr- 6 -  http://prntscr.com/od612s /  -1h-atr 2 -sma20 -  http://prntscr.com/od6g4w - bu daha iyi!!!
#DLTBTC -15m-1tr- 3 - http://prntscr.com/odhwyt - MUTHİŞ!!!!
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
from zigzag import *
from rdp import rdp

class Renko:
	config = ConfigParser()
	config.read("config.ini")
	exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})
	dateformat = '%Y-%m-%d %H:%M'
	dateformat_save = '%Y-%m-%d-%H-%M'


	def __init__(self):
		self.datatype = "local"
		self.interval = "1h"
		self.transaction_fee = 0.00125
		self.initial_balance = 100
		self.results = {}
		self.datatype = "local"
		self.pattern_dir = "/Users/apple/Desktop/dev/projectlife/utils/classification/ma_patterns.json"
		self.atr_coeff_params = [5]#[0.25,0.5,0.75,1,1.25,1.50,1.75,2,2.25,2.5,2.75,3,3.25,3.5,4] #"BTCUSDT"->3, Trx-> 0.5, "CNDBTC"->3.5,  linkbtc->7

		if (self.datatype =="local"):
			self.symbols = ["WTCBTC", "AGIBTC","AEBTC","AIONBTC","ARDRBTC","ARNBTC"]#,"BTTBTC","CDTBTC","CELRBTC","CMTBTC"  "ETHBTC" "BTCUSDT","BCDBTC","DLTBTC","GRSBTC"
			#self.symbols =  ["APPCBTC", "BCPTBTC","BTCUSDT","BCPTBTC","ARNBTC","CNDBTC","DENTBTC","DNTBTC","ENJBTC","GRSBTC","LENDBTC","NEBLBTC"] #
			#self.symbols = ["ADABTC","ADXBTC","AEBTC","AGIBTC","AIONBTC","ALGOBTC","AMBBTC","APPCBTC","ARDRBTC","ARKBTC","ARNBTC","ASTBTC","ATOMBTC","BCDBTC","BCHBTC","BCPTBTC","BLZBTC","BNBBTC","BNTBTC","BQXBTC","BRDBTC", "BTGBTC","BTSBTC","BTTBTC","CDTBTC","CELRBTC","CMTBTC","CNDBTC","CVCBTC","DASHBTC","DATABTC","DCRBTC","DENTBTC","DGDBTC","DLTBTC","DNTBTC","DOCKBTC","EDOBTC","ELFBTC","ENGBTC","ENJBTC","EOSBTC","ERDBTC","ETCBTC","ETHBTC","EVXBTC","FETBTC","FTMBTC","FUELBTC","FUNBTC","GASBTC","GNTBTC","GOBTC","GRSBTC","GTOBTC","GVTBTC","GXSBTC","HCBTC","HOTBTC","ICXBTC","INSBTC","IOSTBTC","IOTABTC","IOTXBTC","KEYBTC","KMDBTC","KNCBTC","LENDBTC","LINKBTC","LOOMBTC","LRCBTC","LSKBTC","LTCBTC","LUNBTC","MANABTC","MATICBTC","MCOBTC","MDABTC","MFTBTC","MITHBTC","MTHBTC","MTLBTC","NANOBTC","NASBTC","NAVBTC","NCASHBTC","NEBLBTC","NEOBTC","NPXSBTC","NULSBTC","NXSBTC","OAXBTC","OMGBTC","ONEBTC","ONGBTC","ONTBTC","OSTBTC","PHBBTC","PIVXBTC","POABTC","POEBTC","POLYBTC","POWRBTC","PPTBTC","QKCBTC","QLCBTC","QSPBTC","QTUMBTC","RCNBTC","RDNBTC","RENBTC","REPBTC","REQBTC","RLCBTC","RVNBTC","SCBTC","SKYBTC","SNGLSBTC","SNMBTC","SNTBTC","STEEMBTC","STORJBTC","STORMBTC","STRATBTC","SYSBTC","TFUELBTC","THETABTC","TNBBTC","TNTBTC","TRXBTC","VETBTC","VIABTC","VIBBTC","VIBEBTC","WABIBTC","WANBTC","WAVESBTC","WPRBTC","WTCBTC","XEMBTC","XLMBTC","XMRBTC","XRPBTC","XVGBTC","XZCBTC","YOYOWBTC","ZECBTC","ZENBTC","ZILBTC","ZRXBTC"]
			#self.symbols = ["ADAUSDT", "ALGOUSDT", "ATOMUSDT", "BATUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "BTTUSDT", "CELRUSDT", "DASHUSDT", "DOGEUSDT", "ENJUSDT", "EOSUSDT", "ERDUSDT", "ETCUSDT", "ETHUSDT", "FETUSDT", "FTMUSDT", "GTOUSDT", "HOTUSDT", "ICXUSDT", "IOSTUSDT", "IOTAUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "MITHUSDT", "NANOUSDT", "NEOUSDT", "NULSUSDT", "OMGUSDT", "ONEUSDT", "ONGUSDT", "ONTUSDT", "PAXUSDT", "QTUMUSDT", "TFUELUSDT", "THETAUSDT", "TRXUSDT", "TUSDUSDT", "USDCUSDT", "USDSUSDT", "USDSBUSDT", "VETUSDT", "WAVESUSDT", "XLMUSDT", "XMRUSDT", "XRPUSDT", "ZECUSDT", "ZILUSDT", "ZRXUSDT"]
		else:
			exchange.load_markets()
			self.symbols = exchange.symbols

	def start_backtest(self):
		for symbol in self.symbols:
			df = self.fetch_symbol(symbol)
			#df = df.tail(50000)
			#df = df.reset_index()
			for atr_coeff in self.atr_coeff_params:
				print(symbol + "-ATR COEFFICENT: "+str(atr_coeff))
				#self.print_df(renko_df) # 2018-08-17 02:30:00
				renko_df = self.populate_renko(df,atr_coeff)
				#renko_df = renko_df.tail(1000)
				#renko_df = renko_df.reset_index()
				#self.divide_cnn_chunks(renko_df,symbol,72)
				#print(df)
				#self.print_df(renko_df)
				#self.identify_pivots(renko_df.close)
				self.evaluate_trades(renko_df,symbol,save_chunks=False,show_trades=True)
				#self.plot_renko(renko_df,symbol,"show")

	def divide_cnn_chunks(self,renko_df,symbol,window_size):
		for i, row in renko_df.iterrows():
			if i > 71:
				fragment = renko_df.iloc[i-window_size:i,:]
				fragment = fragment.reset_index()
				self.plot_renko(fragment,symbol,"save_chunk",buy_index=i,window_size=window_size)

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
		renko_df['sma20'] = talib.SMA(renko_df['close'],20)
		renko_df = self.identify_sr(renko_df)
		return renko_df


	def identify_sr(self,renko_df):
		renko_df['support'] = False
		renko_df['resistance'] = False
		renko_df['sma_res'] = False
		renko_df['sma_sup'] = False
		for i, last_row in renko_df.iterrows():
			prev1,prev2,prev3,prev4,prev5, prev6,prev7,prev8,prev9,prev10,prev11,prev12,prev13,prev14,prev15, prev20  = renko_df.iloc[i-1],renko_df.iloc[i-2],renko_df.iloc[i-3],renko_df.iloc[i-4],renko_df.iloc[i-5],renko_df.iloc[i-6],renko_df.iloc[i-7],renko_df.iloc[i-8],renko_df.iloc[i-9],renko_df.iloc[i-10],renko_df.iloc[i-11],renko_df.iloc[i-12],renko_df.iloc[i-13],renko_df.iloc[i-14],renko_df.iloc[i-15],renko_df.iloc[i-20]
			resistance = (last_row.uptrend == False and prev1.uptrend == True and prev2.uptrend == True and prev1.close > last_row.close and prev1.close > prev2.close)
			support = (last_row.uptrend == True and prev1.uptrend == False and prev2.uptrend == False and prev1.close < last_row.close and prev1.close < prev2.close)
			sma_res = last_row.sma20 < prev5.sma20 and prev9.sma20 < prev10.sma20 and prev10.sma20 > prev11.sma20 and prev15.sma20 > prev20.sma20
			sma_sup = last_row.sma20 > prev5.sma20 and prev9.sma20 > prev10.sma20 and prev10.sma20 < prev11.sma20 and  prev15.sma20 < prev20.sma20
			if support:
				renko_df.at[prev1.name, "support"] = True
			if resistance:
				renko_df.at[prev1.name, "resistance"] = True
			if sma_res:
				renko_df.at[prev10.name, "sma_res"] = True
			if sma_sup:
				renko_df.at[prev10.name, "sma_sup"] = True
		return renko_df

	def identify_pivots(self,X):
		pivots = peak_valley_pivots(X.values, 0.2, -0.2)
		ts_pivots = pd.Series(X, index=X.index)
		ts_pivots = ts_pivots[pivots != 0]
		X.plot(marker=".")
		ts_pivots.plot(style='g-o');
		plt.show()

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
		for i, last_row in renko_df.iterrows():
			current_tick += 1
			buy_patterns = [[False,False,False,True]]
			sell_patterns = [[False,False,True,False,False]]
			if i > 111 and i < len(renko_df) -1:
				current_price = last_row['close']
				prev1,prev2,prev3,prev4,prev5, prev6, prev7, prev8,prev9,prev10,prev11,prev12,prev13,prev14,prev15 = renko_df.iloc[i-1],renko_df.iloc[i-2],renko_df.iloc[i-3],renko_df.iloc[i-4],renko_df.iloc[i-5],renko_df.iloc[i-6],renko_df.iloc[i-7],renko_df.iloc[i-8],renko_df.iloc[i-9],renko_df.iloc[i-10],renko_df.iloc[i-11],renko_df.iloc[i-12],renko_df.iloc[i-13],renko_df.iloc[i-14],renko_df.iloc[i-15]
				cross_5_10_down = last_row.sma5 < last_row.sma10 and prev5.sma10 < prev5.sma5
				sma_20_down = (last_row.sma20 < prev1.sma20)
				fragment = renko_df.iloc[i-32:i+10,:]
				#pattern_up =(last_row.uptrend == True and prev1.uptrend == False)# and prev2.uptrend == True and prev3.uptrend == True and prev4.uptrend == False)# and prev4.uptrend == True and prev5.uptrend == False) ###prev2den oteye bozuluyor
				#brick_sma_up = last_row.close > last_row.sma20 and prev3.close > prev3.sma20 #and prev4.close > pre4.sma20 - prev5.close > pre5.sma20### olmuyor
				#brick_sma_down = last_row.close < last_row.sma20 and prev3.close < prev3.sma20
				#pattern_down =(last_row.uptrend == False and prev1.uptrend == True)# and prev2.uptrend == False and prev3.uptrend == True) ### olmuyor
				#stable_up_all = last_row.sma5 > prev5.sma5 #and last_row.sma10 > prev5.sma10 and last_row.sma20 > prev5.sma20
				#cross_up = last_row.sma5 > last_row.sma10 and prev5.sma10 > prev5.sma5  ### olmuyor
				#cross_up_big =  last_row.sma200 > prev5.sma200 #last_row.sma100 > prev10.sma100 and  ### olmuyor
				#stable_up_sma5 = last_row.sma5 > prev1.sma5 and prev1.sma5 > prev2.sma5 #and prev2.sma5 > prev3.sma5 ### olmuyor
				#stable_up_sma10 = last_row.sma10 > prev1.sma10 and prev1.sma10 > prev2.sma10 and prev2.sma10 > prev3.sma10 ### olmuyor
				#prev_cross_up = prev1.sma5 > prev1.sma10 and prev2.sma5 > prev2.sma10 and prev3.sma5 > prev3.sma10 ### olmuyor

				# zigzag_sma_up = last_row.sma20 > prev10.sma20
				# resistance_array = []
				# top_point = None
				# buy_point= None
				# if zigzag_sma_up:
				# 	fragment = renko_df.iloc[i-100:i-10,:]
				# 	reversed_fragment = fragment.iloc[::-1]
				# 	reversed_fragment  = reversed_fragment.reset_index()
				# 	for ir, row_ir in reversed_fragment.iterrows():
				# 		if row_ir.resistance == True:
				# 			resistance_array.append(row_ir.close)
				# 	sma_sup = reversed_fragment.loc[reversed_fragment['sma_sup'] == True]
				# 	sma_res = reversed_fragment.loc[reversed_fragment['sma_res'] == True]
				# 	for ir, row_ir in reversed_fragment.iterrows():
				# 		if row_ir.sma20 < reversed_fragment.iloc[ir-1].sma20 and row_ir.sma20 > last_row.sma20 and row_ir.sma20 > prev10.sma20:
				# 			top_point = row_ir.sma20
				# 			break
				# 	if len(resistance_array) > 0 and len(sma_sup) > 1 and len(sma_res) >0:
				# 		max_res = max(resistance_array)
				# 		res_sup_check = (sma_res.iloc[0].sma20 > sma_sup.iloc[0].sma20  and sma_res.iloc[0].sma20 > sma_sup.iloc[1].sma20 and sma_sup.iloc[0].sma20 > sma_sup.iloc[1].sma20)
				# 		if res_sup_check and top_point != None and max_res > top_point and last_row.close >= max_res and top_point > reversed_fragment.iloc[-1].sma20:
				# 			#pdb.set_trace()
				# 			buy_point = max_res
				# 			print("buy point" + str(max_res) + "-"+str(last_row.name))

				#zigzag_brick_pattern1 = (last_row.uptrend == True and prev1.uptrend == False and prev2.uptrend == True and prev3.uptrend == True and prev4.uptrend == True and prev5.uptrend == True)
				#zigzag_brick_pattern2 = (last_row.uptrend == True and prev1.uptrend == True and prev2.uptrend == True and prev3.uptrend == False and prev4.uptrend == False and prev5.uptrend == True and prev6.uptrend == True and prev7.uptrend == True and prev8.uptrend == True and prev9.uptrend == True)
				#zigzag_brick_pattern3 = (last_row.uptrend == True and prev1.uptrend == True and prev2.uptrend == True and prev3.uptrend == True and prev4.uptrend == False and prev5.uptrend == False and prev6.uptrend == False and prev7.uptrend == True and prev8.uptrend == True and prev9.uptrend == True and prev10.uptrend == True and prev11.uptrend == True)

				buy_cond = False
				for ix, px in enumerate(buy_patterns):
					for iy, py in enumerate(px):
						check = (renko_df.iloc[i-(iy+1)].uptrend == py)
						if check:
							buy_cond = True

				# sell_cond = False
				# for ix, px in enumerate(sell_patterns):
				# 	for iy, py in enumerate(px):
				# 		check = (renko_df.iloc[i-(iy+1)].uptrend == py)
				# 		if check:
				# 			sell_cond = True


				#buy_cond = zigzag_brick_pattern1 and zigzag_brick_pattern2 or zigzag_brick_pattern3
				#### BUY CONDITIONS #####
				#buy_cond = (buy_point != None and last_row.close >= buy_point)
				#### SELL CONDITIONS #####
				#sell_cond =  sma_20_down # cross_down #and pattern_down# and brick_sma_down#
				sell_cond = (prev2.uptrend == True and prev1.uptrend == False and last_row.uptrend == False)

				if buy_cond and buy_mode and sell_index < i:
					#print(fragment)
					#pdb.set_trace()
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
						print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(last_row.date))
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
						print("SELL: " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(last_row.date) )
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

	def plot_renko(self,df,symbol,type,buy_index=None,window_size=None):
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
		#plt.plot(df.index,df.sma10,color="purple", marker=".",linewidth=5)
		plt.plot(df.index,df.sma10, color="orange")
		if type == "show":
			plt.grid(True)
			fig.suptitle(symbol)
			self.print_df(df)
			plt.show()
		elif type == "save":
			plt.grid(True)
			fig.suptitle(symbol)
			path = "/Users/apple/Desktop/dev/projectlife/data/images/renko/"+symbol+"-buy-"+str(buy_index)+".png"
			fig.savefig(path)
		elif type == "save_chunk":
			frame1 = plt.gca()
			frame1.axes.xaxis.set_ticklabels([])
			frame1.axes.yaxis.set_ticklabels([])
			plt.axis('off')
			plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
			path = "/Users/apple/Desktop/dev/projectlife/data/images/renko/chunks/"+symbol+"-buy-"+str(buy_index)+"-window-"+str(window_size)+".png"
			fig.savefig(path)

def main():
	backtest = Renko()
	backtest.start_backtest()

if __name__ == "__main__":
	main()

