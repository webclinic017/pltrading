"""
===========================================
Bag-of-SFA Symbols in Vector Space (BOSSVS)
===========================================

This example shows how the BOSSVS algorithm transforms a dataset
consisting of time series and their corresponding labels into a
document-term matrix using tf-idf statistics. Each class is represented
as a tfidf vector. For an unlabeled time series, the predicted label is
the label of the tfidf vector giving the highest cosine similarity with
the tf vector of the unlabeled time series.
It is implemented as :class:`pyts.classification.BOSSVS`.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause
import platform
import numpy as np
import matplotlib
import pdb
import json
import pandas as pd
import matplotlib.pyplot as plt
from pyts.classification import BOSSVS
from pyts.classification import SAXVSM
from pyts.datasets import load_gunpoint
from pyts.datasets import load_coffee
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.utils import to_categorical
from keras.preprocessing import sequence
from pathlib import Path
from matplotlib import pyplot as plt
from keras.utils import to_categorical
if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
    matplotlib.use("macOSX")

window_size = 1000
X,Y = [],[]
with open('/Users/apple/Desktop/dev/projectlife/hf/patterns/spike_patterns.json') as json_file:
    patterns = json.load(json_file)
    for pattern in patterns:
        data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/" +  pattern['data'] +"/"+pattern['symbol']+".csv")
        df = pd.DataFrame(data_base)
        df.columns = ['symbol','date','price_change','price_change_percent','last_price','best_bid_price','best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
        fragment = df.iloc[pattern["end"]-window_size+1:pattern["end"]+1,:]
        X.append(fragment.total_traded_quote_asset_volume.values.tolist())
        if pattern['type'] == "spike_true":
            Y.append(1)
            #Y.append([float(0),float(1)])
        else:
            Y.append(0)
            #Y.append([float(1),float(0)])

    X = np.array(X)
    Y = np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


# print("BOSSVSV Fitting...")
# clf =  BOSSVS(word_size=2, n_bins=3, window_size=3)
# print(clf.fit(X_train, y_train))
# print("Predictions:")
# print(clf.predict(X_test))
# print("Real:")
# print(y_test)
# print("Score:")
# a = clf.score(X_test, y_test)
# print("{:10.2f}".format(a))


print("LSTM Fitting...")
verbose, epochs, batch_size = 1, 50, 1
n_timesteps, n_features, n_outputs = X_train.shape[1],1,1
X_train= X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test= X_test.reshape(X_test.shape[0],X_test.shape[1],1)
model = Sequential()
model.add(LSTM(200, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
scores = model.evaluate(X_test, y_test,  batch_size=batch_size, verbose=verbose)
print("Accuracy: %.2f%%" % (scores[1]*100))
probs = model.predict(X_test)
predicted = probs.argmax(axis=1)
print(predicted)

# pdb.set_trace()
# print("ConvLSTM2DLSTM Fitting...")
# verbose, epochs, batch_size = 1, 1, 64

# X_train2= X_train.reshape(X_train.shape[0],X_train.shape[1],1)
# X_test2= X_test.reshape(X_test.shape[0],X_test.shape[1],1)

# n_timesteps, n_features, n_outputs = X_train2.shape[1], X_train2.shape[2], y_train.shape[1]
# n_steps, n_length = 1, 1000
# X_train2 = X_train2.reshape((X_train2.shape[0], 1, 1, n_length, n_features))
# X_test2 = X_test2.reshape((X_test2.shape[0], n_steps, 1, n_length, n_features))
# model = Sequential()
# model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(n_outputs, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train2, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# scores = model.evaluate(X_test2, y_test, batch_size=batch_size, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# probs = model.predict(X_test2)
# predicted = probs.argmax(axis=1)
# print(predicted)




