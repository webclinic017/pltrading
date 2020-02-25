
import scipy.io
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from pathlib import Path
from matplotlib import pyplot as plt
from keras.utils import to_categorical
import pdb

def load_data(direc,dataset):
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
    data_test = np.loadtxt(datadir+'_TEST',delimiter=',')
    X_train, X_test, y_train, y_test = [],[], [], []

    y_train = data_train[:,0]-1
    y_test = data_test[:,0]-1

    for index, x in enumerate(data_train):
        X_train.append(data_train[index][1:74])
    X_train = np.array( X_train)

    for index, x in enumerate(data_test):
        X_test.append(data_test[index][1:74])
    X_test = np.array(X_test)

    return X_train, X_test, y_train, y_test

X_train = []
Y_train = []
X_test = []
Y_test = []



direc = '/Users/apple/Desktop/dev/projectlife/data/UCR'
summaries_dir = '/Users/apple/Desktop/dev/projectlife/data/logs'

"""Load the data"""
X_train,X_test,Y_train,Y_test = load_data(direc,dataset='Gun_Point')

X_train= X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test= X_test.reshape(X_test.shape[0],X_test.shape[1],1)
#Y_train = to_categorical(Y_train,num_classes=1)
#Y_test = to_categorical(Y_test)


print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)

model = Sequential()
model.add(LSTM(200, input_shape=(73, 1)))
model.add(Dense(1, activation='softmax'))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
print(model.summary())

# Train model
history = model.fit(X_train, Y_train, epochs=10, batch_size=1)

# Evaluate model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
probs = model.predict(X_test)
predicted = probs.argmax(axis=1)
pdb.set_trace()

