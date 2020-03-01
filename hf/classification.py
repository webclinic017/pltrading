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
from pyts.datasets import load_gunpoint
from pyts.datasets import load_coffee
if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
	matplotlib.use("macOSX")


middle_window_size = 72
labels = ['neutral', 'pattern']

X_train, Y_train,  X_test, Y_test = [],[],[],[]
train_start_date = pd.to_datetime("2018-09-26 03:30")
test_start_date = pd.to_datetime("2019-04-28 03:30")
with open('/Users/apple/Desktop/dev/projectlife/lstm/strict_patterns2.json') as json_file:
    data = json.load(json_file)
    for elem in data:
        symbol = elem['symbol']
        pattern = elem['patterns']['5m']
        data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_NEOBTC-5m.csv")
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

X_train = np.array(X_train)
y_train = np.array(Y_train)
X_test = np.array(X_test)
y_test = np.array(Y_test)

X_traina, X_testa, y_traina, y_testa = load_gunpoint(return_X_y=True)
#pdb.set_trace()

clf = BOSSVS(window_size=28)

clf.fit(X_train, y_train)
print("Predictions:")
#print(clf.predict(X_test))
print("Real:")
#print(y_test)
print("Score:")
a = clf.score(X_test, y_test)
print("{:10.2f}".format(a))