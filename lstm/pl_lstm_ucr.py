# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:46:39 2017

@author: rob
"""

"""
LSTM for time series classification
Made: 30 march 2016

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""
import numpy as np
import os
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.contrib.rnn import LSTMCell
tf.logging.set_verbosity(tf.logging.ERROR)
#from tensorflow.contrib.rnn import core_rnn

def load_data(direc,dataset):
	direc = '/Users/apple/Desktop/dev/projectlife/data/UCR'
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


def sample_batch(X_train,y_train,batch_size):
	""" Function to sample a batch for training"""
	N,data_len = X_train.shape
	ind_N = np.random.choice(N,batch_size,replace=False)
	X_batch = X_train[ind_N]
	y_batch = y_train[ind_N]
	return X_batch,y_batch


class Model():
	def __init__(self,config):
		num_layers = config['num_layers']
		hidden_size = config['hidden_size']
		max_grad_norm = config['max_grad_norm']
		self.batch_size = config['batch_size']
		sl = config['sl']
		learning_rate = config['learning_rate']
		num_classes = config['num_classes']
		"""Place holders"""
		self.input = tf.placeholder(tf.float32, [None, sl], name = 'input')
		self.labels = tf.placeholder(tf.int64, [None], name='labels')
		self.keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')

		with tf.name_scope("LSTM_setup") as scope:
			def single_cell():
				return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size),output_keep_prob=self.keep_prob)

			cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
			initial_state = cell.zero_state(self.batch_size, tf.float32)

		input_list = tf.unstack(tf.expand_dims(self.input,axis=2),axis=1)
		outputs,_ = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)

		output = outputs[-1]
		self.output = output

		with tf.name_scope("Softmax") as scope:
			with tf.variable_scope("Softmax_params"):
				softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
				softmax_b = tf.get_variable("softmax_b", [num_classes])
			logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.labels,name = 'softmax')
			self.cost = tf.reduce_sum(loss) / self.batch_size
		with tf.name_scope("Evaluating_accuracy") as scope:
			correct_prediction = tf.equal(tf.argmax(logits,1),self.labels)
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			h1 = tf.summary.scalar('accuracy',self.accuracy)
			h2 = tf.summary.scalar('cost', self.cost)

		"""Optimizer"""
		with tf.name_scope("Optimizer") as scope:
		  tvars = tf.trainable_variables()
		  grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),max_grad_norm)   #We clip the gradients to prevent explosion
		  optimizer = tf.train.AdamOptimizer(learning_rate)
		  gradients = zip(grads, tvars)
		  self.train_op = optimizer.apply_gradients(gradients)
		  # Add histograms for variables, gradients and gradient norms.
		  # The for-loop loops over all entries of the gradient and plots
		  # a histogram. We cut of
		  # for gradient, variable in gradients:  #plot the gradient of each trainable variable
		  #       if isinstance(gradient, ops.IndexedSlices):
		  #         grad_values = gradient.values
		  #       else:
		  #         grad_values = gradient
		  #
		  #       tf.summary.histogram(variable.name, variable)
		  #       tf.summary.histogram(variable.name + "/gradients", grad_values)
		  #       tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

		#Final code for the TensorBoard
		self.merged = tf.summary.merge_all()
		self.init_op = tf.global_variables_initializer()
		print('Finished computation graph')

def plot_fragment(data, prediction):
	plt.clf()
	plt.plot(data,label="close", marker="o")
	path = '/Users/apple/Desktop/dev/projectlife/data/images/analysis/test/'+ str(prediction) + "-" + str(data[-1]) + ".png"
	plt.savefig(path)

def backtest():
	SYMBOLS = ["ATOM/BTC",
			  "BCH/BTC",
			  "BCPT/BTC",
			  "BLZ/BTC",
			  "BNB/BTC",
			  "BNT/BTC",
			  "BQX/BTC",
			  "BRD/BTC",
			  "BTG/BTC",
			  "BTS/BTC",
			  "BTT/BTC",
			  "CDT/BTC",
			  "CELR/BTC",
			  "CMT/BTC",
			  "CND/BTC",
			  "CVC/BTC"]

	for symbol in SYMBOLS:
		symbol =  symbol.split("/")[0] + symbol.split("/")[1]
		data_base = read_csv("/Users/apple/Desktop/dev/projectlife/data/binance/Binance_"+symbol+"-"+interval+".csv")
		window_size = 73
		df = DataFrame(data_base, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
		for i, last_row in df.iterrows():
			if i > window_size:
				middle_slide = df.iloc[i-window_size:i+1,:]
				middle_close = middle_slide['close'].values.tolist()
				if last_row.date in train_patterns:
					middle_close.insert(0, "1")
					df_train = df_train.append(pd.Series(middle_close),ignore_index=True)
				else:
					if i % 2500 == 0:
						middle_close.insert(0, "2")
						df_train = df_train.append(pd.Series(middle_close),ignore_index=True)



direc = '/Users/apple/Desktop/dev/projectlife/data/UCR'
summaries_dir = '/Users/apple/Desktop/dev/projectlife/data/logs'

"""Load the data"""
X_train,X_test,y_train,y_test = load_data(direc,dataset='Projectlife')
N,sl = X_train.shape
num_classes = len(np.unique(y_train))

"""Hyperparamaters"""
batch_size = 30
max_iterations = 3000
dropout = 0.8
config = {    'num_layers' :    3,               #number of layers of stacked RNN's
			  'hidden_size' :   120,             #memory cells in a layer
			  'max_grad_norm' : 5,             #maximum gradient norm during training
			  'batch_size' :    batch_size,
			  'learning_rate' : .005,
			  'sl':             sl,
			  'num_classes':    num_classes}

epochs = np.floor(batch_size*max_iterations / N)
print('Train %.0f samples in approximately %d epochs' %(N,epochs))

#Instantiate a model
model = Model(config)

saver = tf.train.Saver()

"""Session time"""
sess = tf.Session() #Depending on your use, do not forget to close the session
writer = tf.summary.FileWriter(summaries_dir, sess.graph)  #writer for Tensorboard
sess.run(model.init_op)
cost_train_ma = -np.log(1/float(num_classes)+1e-9)  #Moving average training cost
acc_train_ma = 0.0
try:
	for i in range(max_iterations):
		X_batch, y_batch = sample_batch(X_train,y_train,batch_size)

		#Next line does the actual training
		cost_train, acc_train,_ = sess.run([model.cost,model.accuracy, model.train_op],feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:dropout})
		cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
		acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
		if i%100 == 1:
			#Evaluate validation performance
			X_batch, y_batch = sample_batch(X_test,y_test,batch_size)
			cost_test, summ,acc_test = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
			print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_test,cost_train_ma,acc_train,acc_test,acc_train_ma))
			#Write information to TensorBoard
			writer.add_summary(summ, i)
			writer.flush()
except KeyboardInterrupt:
	pass
feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0}
#prediction_op = tf.round(y_batch)
#predictions = sess.run(prediction_op, feed_dict=feed_dict)
#print(predictions)

pdb.set_trace()
print("Predictions:",sess.run(tf.math.argmax(model.output, 1), {model.input:X_batch, model.keep_prob:1.0}))
#model.output.eval(feed_dict = {x:X_batch})

# for index, prediction in enumerate(predictions):
# 	plot_fragment(X_batch[index], int(prediction))
# saver.save(sess, 'my_test_model',global_step=1000)
# with tf.Session() as sess:
# 	saver = tf.train.import_meta_graph('my_test_model-1000.meta')
# 	saver.restore(sess,tf.train.latest_checkpoint('./'))
epoch = float(i)*batch_size/N
print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f'%(epoch,acc_test,cost_test))




