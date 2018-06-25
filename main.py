import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import csv

#placeholder: a way to feed data into graphs
#feed_dict: a dictionary to pass numeric values to computational graph

seed = 128
rng = np.random.RandomState(seed)

temp = []
with open('train.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar = '|')
	for row in spamreader:
		temp.append(row)

train_x = np.stack(temp)

temp = []
with open('test.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar = '|')
	for row in spamreader:
		temp.append(row)

test_x = np.stack(temp)

split_size = int(train_x.shape[0]*0.7)
orig = train_x

temp = []
labelTemp = []
train_x = orig[:split_size]
for row in train_x:
	temp.append(row[:-1])
	labelTemp.append(row[-1:])

train_x = np.stack(temp)
train_y = np.stack(labelTemp)

val_x = orig[split_size:]

temp = []
labelTemp = []
for row in val_x:
	temp.append(row[:-1])
	labelTemp.append(row[-1:])

val_x = np.stack(temp)
val_y = np.stack(labelTemp)

#Defining tensorflow variables/instantiating values
input_num_units = 5 * 5
hidden_num_units = 50
output_num_units = 2

epochs = 5
batch_size = 20
learning_rate = 0.01


def indexTrain(start, end):
	return train_x[start * 20: end * 20]

def indexLabel(start, end):
	return train_y[start * 20: end * 20]


x = tf.placeholder(tf.float32, [None, 9])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 1])#This line might be wrong


W1 = tf.Variable(tf.random_normal([9, 20], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([20]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([20, 2], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([2]), name='b2')

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(init_op)
	total_batch = int(len(train_y) / batch_size)
	for epoch in range(epochs):
		avg_cost = 0
		for i in range(total_batch):
			currTrain = indexTrain(i, i + 1)
			currLabel = indexLabel(i, i + 1)
			_, c = sess.run([optimiser, cross_entropy], feed_dict={x: currTrain, y: currLabel})
			avg_cost += c / total_batch
"""
weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):
		avg_cost = 0
		total_batch = int(train.shape[0] / batch_size)

		for i in range(total_batch):
			batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
			_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
"""