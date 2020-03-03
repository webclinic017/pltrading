from keras.utils import to_categorical
from keras.layers import Dense,Dropout,Conv1D,BatchNormalization,MaxPool1D,Flatten
from keras.models import Sequential
from keras.callbacks import History
from keras.layers import LSTM,RNN
import pandas as pd
import numpy as np
import keras
from os import listdir
from sklearn.model_selection import train_test_split
import platform
import pdb
import matplotlib.pyplot as plt
import matplotlib
if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
    matplotlib.use("macOSX")
import json




def normalize(_d, to_sum=True, copy=True):
    # d is a (n x dimension) np array
    d = _d if not copy else np.copy(_d)
    d -= np.min(d, axis=0)
    d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0))
    return d

# Loading train and test data
time_steps=1000
channels=2
batch=10
window_size = 1000

X,Y,PredictX  = [],[],[]
with open('/Users/apple/Desktop/dev/projectlife/hf/patterns/spike_patterns.json') as json_file:
    patterns = json.load(json_file)
    for pattern in patterns:
        data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/" +  pattern['data'] +"/"+pattern['symbol']+".csv")
        df = pd.DataFrame(data_base)
        df.columns = ['symbol','date','price_change','price_change_percent','last_price','best_bid_price','best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
        fragment = df.iloc[pattern["end"]-window_size+1:pattern["end"]+1,:]
        if pattern['type'] == "spike_true":
            X.append([fragment.total_traded_quote_asset_volume.values.tolist(),fragment.last_price.values.tolist()])
            Y.append(1)
        elif pattern['type'] == "spike_false":
            X.append([fragment.total_traded_quote_asset_volume.values.tolist(),fragment.last_price.values.tolist()])
            Y.append(0)
        else:
            PredictX.append([fragment.total_traded_quote_asset_volume.values.tolist(),fragment.last_price.values.tolist()])

    fragment_none = df.iloc[1001:1500,:]
    for i, row in  fragment_none.iterrows():
        middle_slide = df.iloc[i-1000:i,:]
        X.append([middle_slide.total_traded_quote_asset_volume.values.tolist(),middle_slide.last_price.values.tolist()])
        Y.append(0)


X = np.array(X)
X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
Y = np.array(Y)
train_data, test_data, y_train, y_test = train_test_split(X, Y, test_size=0.33)

PredictX = np.array(PredictX)
PredictX = PredictX.reshape(PredictX.shape[0], PredictX.shape[2], PredictX.shape[1])

# path="/Users/apple/Desktop/dev/HAR/"
# features=[x for x in listdir(path+"train/Inertial Signals") if '~' not in x]
# features.sort()
# features1=[x for x in listdir(path+"test/Inertial Signals") if '~' not in x]
# features1.sort()

# y_train=np.loadtxt(path+"train/y_train.txt")-1
# y_test=np.loadtxt(path+"test/y_test.txt")-1

# train_data=np.zeros((y_trainx.shape[0],128,9))
# test_data=np.zeros((y_testx.shape[0],128,9))

# for i in range(len(features)):
#     tr_data=np.loadtxt(path+'/train/Inertial Signals/'+features[i])
#     te_data=np.loadtxt(path+'/test/Inertial Signals/'+features1[i])

#     train_data[:,:,i]=tr_data
#     test_data[:,:,i]=te_data

# print(train_data.shape)
# print(test_data.shape)


# #### Normalizing the data
train_data=(train_data-np.mean(train_data,axis=0)[None,:,:])/np.std(train_data,axis=0)[None,:,:]
test_data=(test_data-np.mean(test_data,axis=0)[None,:,:])/np.std(test_data,axis=0)[None,:,:]
print(train_data.shape)
print(test_data.shape)
print(y_train.shape)
print(y_test.shape)

# Train-Val split
x_train,x_val,y_train,y_val=train_test_split(train_data,y_train,stratify=y_train,random_state=123)
print(y_train.shape)
print(y_train[0])

# #### One-hot encoding of data
y_train=to_categorical(y_train)
y_val=to_categorical(y_val)
y_test=to_categorical(y_test)
print(y_train.shape)

# #### Convolutional Neural Network
history=History()
model=Sequential()
model.add(Conv1D(filters=18,kernel_size=2,strides=1,padding='same',activation='relu',input_shape=(time_steps,channels)))
model.add(MaxPool1D(pool_size=2,strides=2,padding='same'))
model.add(Conv1D(filters=36,kernel_size=2,strides=1,padding='same',activation='relu'))
model.add(MaxPool1D(pool_size=2,strides=2,padding='same'))
model.add(Conv1D(filters=72,kernel_size=2,strides=1,padding='same',activation='relu'))
model.add(MaxPool1D(pool_size=2,strides=2,padding='same'))
model.add(Conv1D(filters=144,kernel_size=2,strides=1,padding='same',activation='relu'))
model.add(MaxPool1D(pool_size=2,strides=2,padding='same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
print(model.summary())
adam=keras.optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=adam)
history=model.fit(x_train,y_train,batch_size=batch,epochs=100,verbose=2,validation_data=(x_val,y_val),shuffle=True)

# This module tests the above creatd CNN model.
[loss,acc]=model.evaluate(test_data,y_test)
print(loss,acc)

pdb.set_trace()
#PredictX = (PredictX-np.mean(PredictX,axis=0)[None,:,:])/np.std(PredictX,axis=0)[None,:,:]

yPredicted = model.predict_proba(PredictX)
predicted = yPredicted.argmax(axis=1)
print(predicted)


probs = model.predict(test_data)
predicted_test = probs.argmax(axis=1)
print(predicted_test)



# Plotting the training accuracy and loss

# plt.plot(history.history['accuracy'],'r-',history.history['val_accuracy'],'b-')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.legend(['train','validation'],loc='bottom right')
# plt.show()

# plt.plot(history.history['loss'],'r-',history.history['val_loss'],'b-')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend(['train','validation'],loc='bottom right')
# plt.show()


# #### Recurrent Neural Network
# history=History()
# model=Sequential()
# model.add(LSTM(units=128,activation='sigmoid',input_shape=(time_steps,channels),dropout=0.3))
# model.add(Dense(6,activation='softmax'))
# print(model.summary())
# rmsprop=keras.optimizers.RMSprop(lr=0.0001)
# adam=keras.optimizers.Adam(lr=0.0001)
# model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
# history=model.fit(x_train,y_train,epochs=100,batch_size=batch,validation_data=(x_val,y_val),verbose=2)

# # Plotting the training accuracy and loss
# plt.plot(history.history['accuracy'],'r-',history.history['val_accuracy'],'b-')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.legend(['train','validation'],loc='bottom right')
# plt.show()

# plt.plot(history.history['loss'],'r-',history.history['val_loss'],'b-')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend(['train','validation'],loc='bottom right')
# plt.show()

# [loss,acc]=model.evaluate(test_data,y_test)
# print(loss,acc)

# # This module tests the above created RNN model.

# # Reading and storing data
# path="/Users/apple/Desktop/dev/HAR/"
# features=[x for x in listdir(path+"train/Inertial Signals") if '~' not in x]
# features1=[x for x in listdir(path+"test/Inertial Signals") if '~' not in x]
# cols=features+["Category"]

# y_train=np.loadtxt(path+"train/y_train.txt").reshape((7352,1))-1
# y_test=np.loadtxt(path+"test/y_test.txt").reshape((2947,1))-1

# x_train=[]
# x_test=[]

# for i in range(len(features)):
#     print(i)
#     vals=np.loadtxt(path+'train/Inertial Signals/'+features[i])
#     vals1=np.loadtxt(path+'test/Inertial Signals/'+features1[i])
#     if i==0:
#         x_train=vals
#         x_test=vals1

#     else:
#         x_train=np.hstack((x_train,vals))
#         x_test=np.hstack((x_test,vals1))

# # Normalize data
# x_train=(x_train-np.mean(x_train,axis=0))/np.std(x_train,axis=0)
# x_test=(x_test-np.mean(x_test,axis=0))/np.std(x_test,axis=0)

# # Stratified train test split
# x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,stratify=y_train,random_state=123)

# x_train=x_train.reshape((x_train.shape[0],9,x_train.shape[1]/9))
# x_val=x_val.reshape((x_val.shape[0],9,x_val.shape[1]/9))
# x_test=x_test.reshape((x_test.shape[0],9,x_test.shape[1]/9))

# np.save('X_train',x_train,allow_pickle=False)
# np.save('X_val',x_val,allow_pickle=False)
# np.save('X_test',x_test,allow_pickle=False)
# np.save('Y_train',y_train,allow_pickle=False)
# np.save('Y_val',y_val,allow_pickle=False)
# np.save('Y_test',y_test,allow_pickle=False)

# # Load data
# x_train,y_train=np.load('X_train.npy',allow_pickle=False),np.load('Y_train.npy',allow_pickle=False)
# x_val,y_val=np.load('X_val.npy',allow_pickle=False),np.load('Y_val.npy',allow_pickle=False)
# x_test,y_test=np.load('X_test.npy',allow_pickle=False),np.load('Y_test.npy',allow_pickle=False)

# # Getting the data in required format
# mlp_x_train=x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# mlp_x_val=x_val.reshape((x_val.shape[0],x_val.shape[1]*x_val.shape[2]))
# mlp_x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

# # One hot encoding of class labels
# y_train=to_categorical(y_train)
# y_val=to_categorical(y_val)
# y_test=to_categorical(y_test)

# # MLP train
# model=Sequential()
# model.add(Dense(1024,activation='relu',input_shape=(128*9,)))
# model.add(Dropout(0.5))
# model.add(Dense(6,activation='softmax'))
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# print(model.summary())
# model.fit(mlp_x_train,y_train,batch_size=64,epochs=10,verbose=2,validation_data=(mlp_x_val,y_val),shuffle=True)

# # MLP test
# [test_loss,test_acc]=model.evaluate(mlp_x_test,y_test)
# print(test_loss,test_acc)

