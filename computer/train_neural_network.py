
__author__ = 'akshay'

import cv2
import tensorflow as tf
import numpy as np
import glob
import sys
from sklearn.model_selection import train_test_split

print 'Loading training data...'
e0 = cv2.getTickCount()

# load training data
image_array = np.zeros((1, 38400))
#image_array = np.zeros((1, 76800))
label_array = np.zeros((1, 3), 'float')
training_data = glob.glob('training_data/*.npz')

# if no data, exit
if not training_data:
    print "No training data in directory, exit"
    sys.exit()

#load training data
for single_npz in training_data:
    with np.load(single_npz) as data:
        train_temp = data['train']
        train_labels_temp = data['train_labels']
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))

X = image_array[1:, :]
y = label_array[1:, :]
#split training data
train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.1)

#initialize nodes of hidden_1_layer
n_nodes_hl1 = 32


n_classes = 3

#create placeholde for INPUT
x = tf.placeholder('float', [1,38400])
#x = tf.placeholder('float', [1,76800])
y = tf.placeholder('float',[1,3])

#define neural network
def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([38400, n_nodes_hl1],stddev=1.0)),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
   
    output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_classes],stddev=1.0)),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(1e-5).minimize(cost)
    
    #epochs
    hm_epochs = 25
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i,j in zip(train,train_labels):
                _, c = sess.run([optimizer, cost], feed_dict={x:[i], y:[j]})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        print (prediction,y)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        for i,j in zip(test,test_labels):
            print('Accuracy:',accuracy.eval({x:[i], y:[j]}))
        for i in test:
            p1=tf.argmax(prediction,1)
            print p1.eval({x:[i]})
            print prediction.eval({x:[i]})
        saver=tf.train.Saver()
        #Save the model after training.
        saver.save(sess,'/home/akshay/Downloads/savedata/project')
train_neural_network(x)
