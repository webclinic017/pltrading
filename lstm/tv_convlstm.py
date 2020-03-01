# convlstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.utils import to_categorical
from matplotlib import pyplot
import pdb

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + '/Users/apple/Desktop/dev/projectlife/data/HAR/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + '/Users/apple/Desktop/dev/projectlife/data/HAR/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    # define model
    verbose, epochs, batch_size = 1, 1, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    #pdb.set_trace()
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

# run the experiment
run_experiment()





# # convlstm model
# from numpy import mean
# from numpy import std
# from numpy import dstack
# from pandas import read_csv
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.layers import TimeDistributed
# from keras.layers import ConvLSTM2D
# from keras.utils import to_categorical
# from matplotlib import pyplot
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, make_scorer, fbeta_score
# from sklearn.preprocessing import MinMaxScaler

# import pdb
# import pandas as pd
# import numpy as np
# import sys
# import json

# # load the dataset, returns train and test X and y elements
# def load_dataset(prefix=''):
#     scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
#     np.set_printoptions(threshold=sys.maxsize)
#     window_size = 72
#     interval = "5m"
#     X_train, Y_train,  X_test, Y_test = [],[],[],[]
#     train_start_date = pd.to_datetime("2018-05-14 00:00")
#     test_start_date = pd.to_datetime("2019-03-14 00:00")
#     with open('/Users/apple/Desktop/dev/projectlife/utils/classification/strict_patterns.json') as json_file:
#         data = json.load(json_file)
#         for elem in data:
#             if elem['symbol'] == "NEOBTC":
#                 patterns = elem['patterns'][interval]
#                 data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+elem['symbol']+"-"+interval+".csv")
#                 df = pd.DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
#                 train_patterns, test_patterns = [],[]
#                 for p in patterns:
#                     end = pd.to_datetime(p["end"])
#                     if end <= test_start_date and end > train_start_date:
#                         train_patterns.append(p["end"])
#                     elif end > test_start_date:
#                         test_patterns.append(p["end"])
#                 for i, row in df.iterrows():
#                     row_date = pd.to_datetime(row.date)
#                     if i > window_size and row_date >train_start_date:
#                         middle_slide = df.iloc[i-window_size:i,:]
#                         middle_close = middle_slide['close'].values.tolist()
#                         if row_date <=test_start_date:
#                             if row.date in train_patterns:
#                                 X_train.append(middle_close)
#                                 Y_train.append([float(0),float(1)])
#                             else:
#                                 if i % 50 == 0:
#                                     X_train.append(middle_close)
#                                     Y_train.append([float(1),float(0)])
#                         else:
#                             if row.date in test_patterns:
#                                 X_test.append(middle_close)
#                                 Y_test.append([float(0),float(1)])
#                             else:
#                                 if i % 500 == 0:
#                                     X_test.append(middle_close)
#                                     Y_test.append([float(1),float(0)])
#                 break

#     X_train = np.array(X_train)
#     Y_train = np.array(Y_train)
#     X_test = np.array(X_test)
#     Y_test = np.array(Y_test)
#     scaler.fit(X_train)
#     scaler.fit(X_test)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
#     X_train = X_train.reshape(X_train.shape[0],1, X_train.shape[1])
#     X_test = X_test.reshape(X_test.shape[0],1, X_test.shape[1])

#     return X_train, Y_train,  X_test, Y_test

# # fit and evaluate a model
# def evaluate_model(trainX, trainy, testX, testy):
#     # define model
#     verbose, epochs, batch_size = 1, 1, 64
#     n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
#     # reshape into subsequences (samples, time steps, rows, cols, channels)
#     n_steps, n_length = 2, 36
#     trainX = trainX.reshape(trainX.shape[0], 72, 1, 1, 1)
#     testX = testX.reshape(testX.shape[0], 72, 1, 1, 1)

#    # trainX = trainX.reshape(trainX.shape[0], n_steps, 1, n_length, n_features)
#     #testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
#     # define model
#     pdb.set_trace()
#     model = Sequential()
#     model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(72, 1, 1, 1)))
#     model.add(Dropout(0.5))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # fit network

#     model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
#     # evaluate model
#     _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
#     probs = model.predict(testX,batch_size=batch_size) #,
#     predicted = probs.argmax(axis=1)
#     print(predicted)
#     labels = ['neutral', 'pattern']

#     confusion_matrix = pd.crosstab(pd.Series(y_index), pd.Series(predicted))
#     confusion_matrix.index = [labels[i] for i in confusion_matrix.index]
#     confusion_matrix.columns = [labels[i] for i in confusion_matrix.columns]
#     confusion_matrix.reindex(columns=[l for l in labels], fill_value=0)
#     print(confusion_matrix)
#     return accuracy


# # summarize scores
# def summarize_results(scores):
#     print(scores)
#     m, s = mean(scores), std(scores)
#     print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# # run an experiment
# def run_experiment(repeats=2):
#     # load data
#     trainX, trainy, testX, testy = load_dataset()
#     # repeat experiment
#     scores = list()
#     for r in range(repeats):
#         score = evaluate_model(trainX, trainy, testX, testy)
#         score = score * 100.0
#         print('>#%d: %.3f' % (r+1, score))
#         scores.append(score)
#     # summarize results
#     summarize_results(scores)

# # run the experiment
# run_experiment()