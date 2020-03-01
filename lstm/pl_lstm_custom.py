import sys
import os
import numpy as np
import pandas as pd
import pdb
import json
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical

np.random.seed(2)
sys.path.insert(0, os.path.abspath('../..'))
middle_window_size = 72
interval = "5m"
labels = ['neutral', 'pattern']
modelname = 'my_bestmodel.h5'
resultpath = '/Users/apple/Desktop/dev/projectlife/lstm/results/models'
model_path = os.path.join(resultpath,modelname)
outputfile = os.path.join(resultpath, 'modelcomparison.json')
np.set_printoptions(threshold=sys.maxsize)
X_train, Y_train,  X_test, Y_test = [],[],[],[]
train_start_date = pd.to_datetime("2018-09-27 00:00")
test_start_date = pd.to_datetime("2019-02-09 00:00")
with open("/Users/apple/Desktop/dev/projectlife/lstm/strict_patterns2.json") as json_file:
    data = json.load(json_file)
    for elem in data:
        symbol = elem['symbol']
        try:
            if symbol == "NEOBTC":
                pattern = elem['patterns'][interval]
                data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
                df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df['pattern'] = "none"
                df['symbol'] = symbol
                train_patterns, test_patterns = [],[]
                for p in pattern:
                    end = pd.to_datetime(p["end"])
                    if end <= test_start_date and end > train_start_date:
                        train_patterns.append(p["end"])
                    elif end > test_start_date:
                        test_patterns.append(p["end"])
                for i, row in df.iterrows():
                    row_date = pd.to_datetime(row.date)
                    if i > middle_window_size and row_date >train_start_date:
                        middle_slide = df.iloc[i-middle_window_size:i,:]
                        middle_close = middle_slide['close'].values.tolist()
                        if row_date <=test_start_date and row_date > train_start_date:
                            X_train.append(middle_close)
                            if row.date in train_patterns:
                                Y_train.append(1)
                            else:
                                Y_train.append(0)
                        elif row_date > test_start_date:
                            X_test.append(middle_close)
                            if row.date in test_patterns:
                                Y_test.append(1)
                            else:
                                Y_test.append(0)
                break
        except:
            print(symbol +" pattern not found")

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
Y_train = Y_train.reshape(Y_train.shape[0],1)

X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
Y_test = Y_test.reshape(Y_test.shape[0],1)

verbose, epochs, batch_size = 1, 5, 72
n_timesteps, n_features, n_outputs = 72,1,1

model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network

model.fit(X_train,Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(X_test,Y_test, batch_size=batch_size, verbose=1)

print("Accuracy: %.2f%%" % (accuracy*100))
probs = model.predict(X_test,batch_size=batch_size)
predicted = probs.argmax(axis=1)
print(predicted)