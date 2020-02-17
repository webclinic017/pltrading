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
import glob
import os
import argparse
from binance.enums import *
from binance.client import Client
from binance.websockets import BinanceSocketManager
import logging
from twisted.internet import task, reactor
import requests
from pymongo import MongoClient
import telegram

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:  %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Manualbot:

	def __init__(self,args):
		self.buy_mode = True
		self.args = args
		self.target_profits = [50,30,20]
		self.targets_reached = [False,False,False]
		self.stoploss_reached = False
		config = ConfigParser()
		config.read("config.ini")
		self.config = config

	def run(self):
		self.client = Client(api_key=self.config['BINANCE']['KEY'], api_secret=self.config['BINANCE']['SECRET'])
		if self.args.mode == "trading":
			logger.info("START TRADE | symbol: {}, btc amount: {}, targets: {}, stoploss price: {}, trailing: {}, trailing price: {}".format(self.args.symbol, self.args.btc_amount, self.args.targets, self.args.immediate_stoploss,  self.args.use_trailing, self.args.trailing_stoploss))
			bm = BinanceSocketManager(self.client)
			self.conn_key = bm.start_symbol_ticker_socket(self.args.symbol, self.process_message)
			bm.start()
		elif self.args.mode == "analysis":
			alltickers = self.client.get_ticker()
			interval = "1h"
			exchange = ccxt.binance({'apiKey': self.config['BINANCE']['KEY'], 'secret': self.config['BINANCE']['SECRET']})
			for ticker in alltickers:
				if float(ticker['priceChangePercent']) > 2 and ("BTC" in ticker['symbol']):
					data_base = exchange.fetch_ohlcv(ticker['symbol'].split("BTC")[0]+"/BTC", interval,limit=100)
					df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close','volume'])
					df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
					self.save_analysis(df,ticker['symbol'],interval, ticker['priceChangePercent'])
		elif self.args.mode == "hamster":
			mongo_client = MongoClient('mongodb://localhost:27017/')
			db = mongo_client.projectlife
			self.previous_response ="initial"
			timeout = 30
			exchanges=dict()
			exchanges['binance'] = ccxt.binance({'apiKey': self.config['BINANCE']['KEY'], 'secret': self.config['BINANCE']['SECRET']})
			exchanges['kucoin'] = ccxt.kucoin({'apiKey': self.config['KUCOIN']['KEY'], 'secret': self.config['KUCOIN']['SECRET']})
			exchanges['bittrex'] = ccxt.bittrex({'apiKey': self.config['BITTREX']['KEY'], 'secret': self.config['BITTREX']['SECRET']})
			exchanges['poloniex'] = ccxt.poloniex()

			def doWork():
				responses = []
				try:
					url = 'https://www.mininghamster.com/api/v2/'+self.config['HAMSTER']['API']
					responses = requests.get(url).json()
					if len(responses)> 0:
						for response in responses:
							symbol = response['market'].split("BTC-")[1] + "/BTC"
							bid,ask = self.get_prices(exchanges[response['exchange']], symbol)
							if response['signalmode'] == "buy":
								result_buy = db.hamster.find_one({"signalID": response['signalID'], "signalmode": "buy"})
								if result_buy == None:
									response['buy_price'] = ask
									db.hamster.insert_one(response)
									self.send_telegram(str(response))
							elif response['signalmode'] == "sell":
								result_sell = db.hamster.find_one({"signalID": response['signalID'], "signalmode": "sell"})
								if result_sell == None:
									result_buy = db.hamster.find_one({"signalID": response['signalID'], "signalmode": "buy"})
									if result_buy != None:
										response['sell_price'] = bid
										response['profit'] = self.pct_change(result_buy['buy_price'],bid)
										db.hamster.insert_one(response)
										self.send_telegram(str(response))


				except BaseException as e:
					print(e)
				pass

			lx = task.LoopingCall(doWork)
			lx.start(timeout)
			reactor.run()
		elif self.args.mode == "datacollect":
			client = MongoClient('mongodb://localhost:27017/')
			db = client.projectlife
			self.db_collection = db[self.args.symbol]
			bm = BinanceSocketManager(self.client)
			self.conn_key = bm.start_symbol_ticker_socket(self.args.symbol, self.process_datacollect_message)
			bm.start()

	def get_prices(self, exchange, symbol):
		orderbook = exchange.fetch_order_book(symbol)
		bid = orderbook['bids'][0][0] if len (orderbook['bids']) > 0 else None
		ask = orderbook['asks'][0][0] if len (orderbook['asks']) > 0 else None
		return bid,ask

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

	def create_order(self,symbol,side,type,quantity,price, actual_type):
		try:
			order = self.client.create_test_order(symbol=symbol,side=side,type=type,timeInForce='GTC',quantity=quantity,price=price)
			print(order)
			if order == {}:
			#if order["status"] == "NEW":
				log = "{} ORDER FILLED | symbol: {}, type: {}, quantity: {}, price: {}, actual_type: {}".format(side, symbol,type,str(quantity),price,actual_type)
				logger.info(log)
				self.send_telegram(log)
				return True
			else:
				log = "{} ORDER NOT FILLED | symbol: {}, type: {}, quantity: {}, price: {}, actual_type: {}".format(side, symbol,type,str(quantity),price,actual_type)
				logger.info(log)
				self.send_telegram(log)
				return False
		except BaseException as e:
			log = 'Error occured when creating order: '+ str(e)
			logger.error(log)
			self.send_telegram(log)
			return False

	def send_telegram(self,msg):
		bot = telegram.Bot(token=self.config['TELEGRAM']['KEY'])
		bot.sendMessage(chat_id=self.config['TELEGRAM']['CHATID'], text=msg)


	def save_analysis(self, df, symbol,interval, percentchange):
		last_date = df.iloc[-1].date
		df["date"] = df.index.values
		plt.cla()
		fig,ax = plt.subplots(figsize = (16,16))
		candlestick_ohlc(ax, df.values, width=0.6, colorup='green', colordown='red' )
		df['sma5'] = talib.SMA(df['close'],5)
		df['sma8'] = talib.SMA(df['close'],8)
		df['sma13'] = talib.SMA(df['close'],13)
		plt.plot(df.index,df.sma5, color="orange")
		plt.plot(df.index,df.sma8, color="blue")
		plt.plot(df.index,df.sma13, color="red")
		fig.suptitle(symbol+"-"+interval)
		fn = "/Users/apple/Desktop/dev/projectlife/data/images/analysis/"+symbol+"-"+percentchange+"%-"+interval+"-"+last_date+".png"
		plt.savefig(fn, dpi=100)

	def stop_trading(self):
		bm = BinanceSocketManager(self.client)
		bm.stop_socket(self.conn_key)
		bm.close()
		reactor.stop()


	def process_message(self,msg):
		if msg['e'] == 'error':
			print("error on websocket. Restarting")
			bm.stop_socket(self.conn_key)
			bm.start()
		else:
			if self.buy_mode == True:
				ask_price = msg['a']
				self.buy_quantity = int(self.args.btc_amount / float(ask_price))
				self.latest_quantity = self.buy_quantity
				response = self.create_order(self.args.symbol,"BUY",'LIMIT',self.buy_quantity,ask_price, "IMMEDIATE BUY")
				if response == True:
					self.buy_mode = False
			else:
				bid_price = float(msg['b'])
				trl_stoploss_price = bid_price - self.args.trailing_stoploss
				imm_stoploss_price = self.args.immediate_stoploss

				if imm_stoploss_price >= bid_price:
					imm_stoploss_price = "{:.8f}".format(bid_price)
					response = self.create_order(self.args.symbol,"SELL",'LIMIT',self.latest_quantity,imm_stoploss_price, "IMMEDIATE STOPLOSS")
					if response == True:
						self.stop_trading()

				if (trl_stoploss_price >= bid_price and self.args.use_trailing == True and self.targets_reached[0] == True):
					trl_stoploss_price = "{:.8f}".format(bid_price)
					response = self.create_order(self.args.symbol,"SELL",'LIMIT',self.latest_quantity,trl_stoploss_price, "TRAILING STOPLOSS")
					if response == True:
						self.stop_trading()

				for index, target_price in enumerate(self.args.targets):
					target_quantity = int((self.buy_quantity*self.target_profits[index])/100)
					if bid_price >= target_price and self.targets_reached[index] == False:
						target_price = "{:.8f}".format(target_price)
						response = self.create_order(self.args.symbol,"SELL",'LIMIT',target_quantity,target_price, "TAKE PROFIT TARGET "+ str(index+1))
						if response == True:
							self.targets_reached[index] = True
							self.latest_quantity = self.latest_quantity - target_quantity
							break

				if False not in self.targets_reached:
					self.stop_trading()



def main():
	parser = argparse.ArgumentParser(description='\n projetlife manual trading bot')
	parser.add_argument('--symbol', type=str, required=False,                default="ONTBTC",                           help='symbol to be traded')
	parser.add_argument('--btc_amount', type=float, required=False,          default=0.001,                              help='BTC amount to buy')
	parser.add_argument('--targets', nargs='+', required=False,              default=[0.000533,0.000552,0.000575],  help='targets')
	parser.add_argument('--immediate_stoploss', type=float, required=False,   default=0.000509,   						 help='immd stoploss price')
	parser.add_argument('--trailing_stoploss', type=float, required=False,    default=0.000005,                        help='trailing stoploss price')
	parser.add_argument('--use_trailing', type=bool, required=False,         default=True,                              help='trailing stoploss or not')
	parser.add_argument('--mode', type=str, required=False, default="datacollect",											 help='analyze or trade')
	args = parser.parse_args()
	manualbot = Manualbot(args)
	manualbot.run()

if __name__ == "__main__":
	main()


# self.send_telegram(response)
# symbol = response[0]['market'].split("BTC-")[1] + "BTC"
# info = [x for x in  self.client.get_ticker() if x['symbol'] == symbol]
# free_quantity = int(float(self.client.get_asset_balance(asset=response[0]['market'].split("BTC-")[1])['free']))
# if response[0]['signalmode'] == "buy":
# 	buy_quantity = int(self.args.btc_amount / float(info[0]['askPrice']))
# 	if buy_quantity > 0 and free_quantity ==0:
# 		self.create_order(symbol,"BUY",'LIMIT',buy_quantity,info[0]['askPrice'], "IMMEDIATE BUY")
# 	else:
# 		log = "BUY ORDER NOT FILLED| symbol: {}, buy quantity: {}, free quantity: {}".format(symbol,str(buy_quantity),str(free_quantity))
# 		logger.error(log)
# 		self.send_telegram(log)
# elif response[0]['signalmode'] == "sell":
# 	if free_quantity > 0:
# 		self.create_order(symbol,"SELL",'LIMIT',free_quantity,info[0]['bidPrice'], "IMMEDIATE SELL")
# 	else:
# 		log = "SELL ORDER NOT FILLED BECAUSE OF QUANTITY | symbol: {}, free_quantity: {}".format(symbol,str(free_quantity))
# 		logger.error(log)
# 		self.send_telegram(log)
