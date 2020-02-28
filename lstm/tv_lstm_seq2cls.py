#### ÇALIŞMIYOR. BULDUGU BU. CONFUSİON MATRİXİ YARATAMIYOR.
#neutral    87305
#pattern        4

import sys
import os
import numpy as np
import pandas as pd
import pdb
import json
import pandas as pd
from mcfly import modelgen, find_architecture, storage
from keras.models import load_model
np.random.seed(2)
sys.path.insert(0, os.path.abspath('../..'))
middle_window_size = 72
batch_size = 128
interval = "5m"
labels = ['neutral', 'pattern']
modelname = 'my_bestmodel.h5'
resultpath = '/Users/apple/Desktop/dev/projectlife/lstm/results/models'
model_path = os.path.join(resultpath,modelname)
outputfile = os.path.join(resultpath, 'modelcomparison.json')
np.set_printoptions(threshold=sys.maxsize)
X_train, Y_train,  X_test, Y_test = [],[],[],[]
train_start_date = pd.to_datetime("2018-09-26 03:30")
test_start_date = pd.to_datetime("2019-04-28 03:30")
with open('/Users/apple/Desktop/dev/projectlife/lstm/strict_patterns2.json') as json_file:
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
                        if row_date <=test_start_date:
                            X_train.append(middle_close)
                            if row.date in train_patterns:
                                Y_train.append([float(0),float(1)])
                            else:
                                Y_train.append([float(1),float(0)])
                        else:
                            X_test.append(middle_close)
                            if row.date in test_patterns:
                                Y_test.append([float(0),float(1)])
                            else:
                                Y_test.append([float(1),float(0)])
                break
        except:
            print(symbol +" pattern not found")
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

num_classes = Y_train.shape[1]
models = modelgen.generate_models(X_train.shape,  number_of_classes=num_classes, number_of_models = 1)
histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train,Y_train, X_test, Y_test,models,nr_epochs=1,subset_size=300, verbose=True, outputfile=outputfile)
print('Details of the training process were stored in ',outputfile)

modelcomparisons = pd.DataFrame({'model':[str(params) for model, params, model_types in models],
					   'train_acc': [history.history['accuracy'][-1] for history in histories],
					   'train_loss': [history.history['loss'][-1] for history in histories],
					   'val_accuracy': [history.history['val_accuracy'][-1] for history in histories],
					   'val_loss': [history.history['val_loss'][-1] for history in histories]
					   })

modelcomparisons.to_csv(os.path.join(resultpath, 'modelcomparisons.csv'))
modelcomparisons
best_model_index = np.argmax(val_accuracies)
best_model, best_params, best_model_types = models[best_model_index]
print('Model type and parameters of the best model:')
print(best_model_types)
print(best_params)

## Training on Train Data
nr_epochs = 1
history = best_model.fit(X_train, Y_train ,epochs=nr_epochs)
# best_model.save(model_path)
# model_path = resultpath + "/my_bestmodel.h5"
# pdb.set_trace()
# model_reloaded = load_model(model_path)
# model_reloaded.get_weights()

## Test on Validation Data
#probs = model_reloaded.predict(X_test,batch_size=batch_size)
probs = best_model.predict(X_test,batch_size=batch_size)
predicted = probs.argmax(axis=1)
y_index = Y_test.argmax(axis=1)
confusion_matrix = pd.crosstab(pd.Series(y_index), pd.Series(predicted))
confusion_matrix.index = [labels[i] for i in confusion_matrix.index]
confusion_matrix.columns = [labels[i] for i in confusion_matrix.columns]
confusion_matrix.reindex(columns=[l for l in labels], fill_value=0)
print(confusion_matrix)
#pdb.set_trace()

## Test on Test Data
#model_reloaded.compile(loss='mse', optimizer='adam')
#score_test = model_reloaded.evaluate(X_test, Y_test, verbose=True)
#print('Score of best model: ' + str(score_test))




