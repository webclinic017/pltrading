import tensorflow as tf
import numpy as np
import time

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

direc = '/Users/apple/Desktop/dev/projectlife/data/UCR'
summaries_dir = '/Users/apple/Desktop/dev/projectlife/data/logs'

"""Load the data"""
trainX, testX, trainy, testy = load_data(direc,dataset='Projectlife')
testy = testy.reshape(testy.shape[0], 1)

# trainX = np.array(trainX)
# trainy = np.array(trainy)
# trainy = trainy.reshape(trainy.shape[0], 1)
# testX = np.array(testX)
# testy = np.array(testy)
# print (trainX.shape)
# print (trainy.shape)
# testX = testX.reshape(testX.shape[0], 130)
# testy = testy.reshape(testy.shape[0], 1)
# print (testX.shape)
# print (testy.shape)
n_nodes_hl1 = 256
n_nodes_hl2 = 256
n_nodes_hl3 = 256

n_classes = 1

batch_size = 100


# Matrix = h X w
X = tf.placeholder('float', [None, len(trainX[0])])
y = tf.placeholder('float')



def model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([trainX.shape[1], n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.sigmoid(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train(x):

    pred = model(x)
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
    loss = tf.reduce_mean(tf.square(pred - y))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    epochs = 1000

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print ('Beginning Training \n')
        for e in range(epochs):
            timeS = time.time()
            epoch_loss = 0
            i = 0
            while i < len(trainX):
                start = i
                end = i + batch_size
                batch_x = np.array(trainX[start:end])
                batch_y = np.array(trainy[start:end])
                _, c = sess.run([optimizer, loss], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            done = time.time() - timeS
            print ('Epoch', e + 1, 'completed out of', epochs, 'loss:', epoch_loss, "\nTime:", done, 'seconds\n')
        correct = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
        acc = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy:", acc.eval({x:testX, y:testy}))
        print("Predictions:",sess.run(tf.math.argmax(pred, 1), {x:testX}))
        #predictions = pred.eval(feed_dict = {x:testX})

train(X)
