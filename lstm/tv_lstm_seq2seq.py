from math import sqrt
from dateutil import parser
from configparser import ConfigParser
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras.layers
from keras.models import Sequential
from keras.layers import concatenate, Dense, Flatten, LSTM, RepeatVector, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
import pdb
import matplotlib.pyplot as plt
import numpy as np
import ccxt

SYMBOLS = ["LTC/BTC"] #["STORM/BTC","ZRX/BTC","DENT/BTC", "DLT/BTC","AST/BTC","ETH/BTC","LTC/BTC","NEO/BTC","XMR/BTC", "RVN/BTC","LINK/BTC","BNB/BTC", "HOT/BTC","WTC/BTC","TRX/BTC","ENJ/BTC","QKC/BTC","VET/BTC","ONT/BTC","GO/BTC","ZEC/BTC","REP/BTC"]
feature_count = 1
epochs = 1
n_inputs = [10, 20] # 30, 40, 50
normalize = True
transaction_fee = 0.00125
initial_balance = 100
BUY, SELL, HOLD = 0, 1, 2
test_size = 0.2
time_window = 10
ohlcv_interval="1h"
batch_size = 256 #16 1 256
results = {}
config = ConfigParser()
scaler = MinMaxScaler(feature_range=(0, 1))
config.read("config.ini")
exchange = ccxt.binance({'apiKey': config['BINANCE']['KEY'], 'secret': config['BINANCE']['SECRET']})


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(predicted.shape[1]): #actual.shape[1]
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(predicted.shape[0]):
		for col in range(predicted.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (predicted.shape[0] * predicted.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=1):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return np.array(X), np.array(y)

# train the model
def build_model(train,n_input,symbol,interval):
	sym = symbol.split("/")
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='linear')))
	model.add(TimeDistributed(Dense(1, activation='linear')))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
	# model_json = model.to_json()
	# model_path = "./results/models/seq2seq-lstm-model-"+sym[0]+sym[1]+"-"+interval+"-"+str(n_input)+".json"
	# weights_path = "./results/weights/seq2seq-lstm-weights-"+sym[0]+sym[1]+"-"+interval+"-"+str(n_input)+".h5"
	# with open(model_path, "w") as json_file:
	# 	json_file.write(model_json)
	# model.save_weights(weights_path)
	return model


def equal_train_test(df):

	return train, test

# evaluate a single model
def  train_evaluate_model(n_input, symbol,interval):
	sym = symbol.split("/")
	df = read_csv("/Users/apple/Desktop/dev/projectlife/data/full/Binance_"+sym[0]+sym[1]+"-"+interval+".csv",  index_col=0, header=0)
	pdb.set_trace()
	if feature_count == 1:
		data = df.iloc[:,3]

	if normalize==True:
		data = scaler.fit_transform(data.values.reshape(data.values.shape[0]*feature_count,1))
	else:
		data = data.values.reshape(data.values.shape[0]*feature_count,1)

	train, test = train_test_split(data, test_size=test_size,shuffle=False)

	train = np.array(np.split(train,int(len(train)/time_window)))
	test = np.array(np.split(test,int(len(test)/time_window)))
	# fit model
	model = build_model(train, n_input,symbol,interval)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	timeline = list()
	actual_prices = list()
	for i in range(len(test)):
		yhat_sequence, predict_date, actual_price = forecast(model, history, n_input, df)
		timeline.append(predict_date)
		predictions.append(yhat_sequence)
		actual_prices.append(actual_price)
		history.append(test[i, :])
	return timeline, predictions, actual_prices

# make a forecast
def forecast(model, history, n_input, df):
	# flatten data
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into [1, n_input, n]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	predict_day = df.iloc[len(data),:]
	predict_date = parser.parse(predict_day.name)
	actual_price = predict_day.close
	# we only want the vector forecast
	yhat = yhat[0][0][0]
	return yhat, predict_date, actual_price

def evaluate_trades(df,symbol):
	df = df.set_index("timeline")
	df_pct = df.pct_change()
	df_pct['current_price'] =  df['actuals']#.shift(1)
	balance = initial_balance
	trade_count = 0
	profit = 0
	action = HOLD
	trade_history = []
	current_tick = 0
	buy_mode = True
	for i, row in df_pct.iterrows():
		buy_signal = row['predictions'+str(n_inputs[0])] > 0 and row['predictions'+str(n_inputs[1])] > 0 #and row['predictions0'] > 0
		sell_signal = row['predictions'+str(n_inputs[0])] < 0 and row['predictions'+str(n_inputs[1])] < 0 #and row['predictions30'] < 0
		current_price = row['current_price']
		if buy_signal == True and buy_mode == True:
			action = BUY
			entry_price =  current_price
			quantity = balance / entry_price
			buy_mode = False
			print("##### TRADE " +  str(trade_count) + " #####")
			print("BUY: " + str(quantity) + " " +symbol+" for "+ str(entry_price) + " at " +  str(row.name)  )
		elif sell_signal == True and buy_mode == False:
			action = SELL
			exit_price =  current_price
			profit = ((exit_price - entry_price)/entry_price + 1)*(1-transaction_fee)**2 - 1
			balance = balance * (1.0 + profit)
			entry_price = 0
			trade_count += 1
			buy_mode = True
			print("SELL: " + str(quantity) + " " +symbol+" for "+ str(exit_price)  + " at " +  str(row.name) )
			print("PROFIT: " + str(profit))
			print("BALANCE: " + str(balance))
			print("==================================")
		else:
			action = HOLD

		current_tick += 1
		trade_history.append((action, current_tick, current_price, balance, profit))

		if (current_tick > len(df)-1):
			results[symbol] = {'balance':np.array([balance]), "trade_history":trade_history, "trade_count":trade_count }
			print("**********************************")
			print("TOTAL BALANCE FOR "+symbol +": "+ str(balance))
			print("**********************************")
			plot_buy_sell(trade_history)

	if (symbol == SYMBOLS[-1]):
		print("#########FINAL BALANCES#####################")
		final_balance = 0
		for symbol in SYMBOLS:
			balance = results[symbol]['balance'][0]
			print(symbol + ": "+str(balance))
			final_balance +=balance
		print("============================================")
		print("FINAL BALANCE: " + str(final_balance))



def plot_actual_pred():
	ax = plt.gca()
	if n_input == n_inputs[0]:
		df = pd.DataFrame({'timeline': timeline ,'actuals':actual_prices, pred_name:predictions})
		#df.plot(kind='line',x='timeline',y='actuals', linewidth=4,color='skyblue', ax=ax)
		#df.plot(kind='line',x='timeline', marker="o", linewidth=2, y=pred_name,ax=ax)
	else:
		df[pred_name] = predictions
		#df.plot(kind='line',x='timeline', marker="o", linewidth=2, y=pred_name,ax=ax)

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

def train_and_evaluate():
	for symbol in SYMBOLS:
		for n_input in n_inputs:
			timeline, predictions, actual_prices = train_evaluate_model(n_input, symbol,ohlcv_interval)
			pred_name = 'predictions'+str(n_input)
			if n_input == n_inputs[0]:
				df = pd.DataFrame({'timeline': timeline ,'actuals':actual_prices, pred_name:predictions})
			else:
				df[pred_name] = predictions
		evaluate_trades(df,symbol)

def backtest():
	for symbol in SYMBOLS:
		data_base = exchange.fetch_ohlcv(symbol, ohlcv_interval,limit=1000)
		df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		df["date"] = pd.to_datetime(df["date"], unit = 'ms').dt.strftime('%Y-%m-%d %H:%M')
		df.set_index('date')
		df.replace({0: np.nan}, inplace=True)
		pdb.set_trace()
		df['price'] = df[['open', 'high', 'low', 'close']].mean(axis=1)
		df['price_change'] = df['price'].pct_change()
		input_data = sc.fit_transform(df[['price_change', 'volume_change', 'volatility', 'convergence', 'predisposition']])
		output_data = input_data[:, 0]
		mean = np.mean(output_data, axis=0)
		last_change = output_data[-1] - mean
		predict_change = model.predict(np.array([input_data[-STEP_SIZE:]]), batch_size=1)[0][0] - mean
		if last_change < 0 < .2 < predict_change:
			log('{} ML BUY'.format(symbol.symbol))
			rd.publish('ml_buy', symbol.symbol)
			return True
		elif last_change > 0 > -.1 > predict_change:
			log('{} ML SELL'.format(symbol.symbol))
			rd.publish('ml_sell', symbol.symbol)
			return True
		return False

if __name__ == '__main__':
	train_and_evaluate()
	#backtest()
