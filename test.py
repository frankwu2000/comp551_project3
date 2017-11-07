# handle input data
import numpy as np

# head -n 10000 train_x.csv > train_x_small.csv
features = np.loadtxt("train_x.csv", delimiter=",") # load from text 
labels = np.loadtxt("train_y.csv", delimiter=",") 

print ("done input data")
# doing the split dataset
features_input = features/255
features_set = np.where(features > 252, 1, 0)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
from sklearn import preprocessing
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

lb = preprocessing.LabelBinarizer()
lb.fit(classes)
lb.classes_
label_one_hot = lb.transform(labels).astype(float)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_set,label_one_hot, test_size=0.20, random_state=None)

# reshape data
X = X_train.reshape([-1,64,64,1])
testX = X_test.reshape([-1,64,64,1])
# X.shape

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# # Data loading and preprocessing
# import tflearn.datasets.mnist as mnist
# X, Y, testX, testY = mnist.load_data(one_hot=True)
# X = X.reshape([-1, 28, 28, 1])
# testX = testX.reshape([-1, 28, 28, 1])

# # Building convolutional network
network = input_data(shape=[None, 64, 64, 1], name='input')
network = conv_2d(network, 32, 2, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 128, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 40, activation='softmax')
network = regression(network, optimizer='momentum', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

# alexit net
# network = input_data(shape=[None, 64, 64, 1], name='input')
# network = conv_2d(network, 96, 11, strides=4, activation='relu')
# network = max_pool_2d(network, 3, strides=2)
# network = local_response_normalization(network)
# network = conv_2d(network, 256, 5, activation='relu')
# network = max_pool_2d(network, 3, strides=2)
# network = local_response_normalization(network)
# network = conv_2d(network, 384, 3, activation='relu')
# network = conv_2d(network, 384, 3, activation='relu')
# network = conv_2d(network, 256, 3, activation='relu')
# network = max_pool_2d(network, 3, strides=2)
# network = local_response_normalization(network)
# network = fully_connected(network, 4096, activation='tanh')
# network = dropout(network, 0.5)
# network = fully_connected(network, 4096, activation='tanh')
# network = dropout(network, 0.5)
# network = fully_connected(network, 40, activation='softmax')
# network = regression(network, optimizer='momentum',
#                      loss='categorical_crossentropy',
#                      learning_rate=0.001, name='target')

# Training
model = tflearn.DNN(network, best_checkpoint_path='modelbest.tfl.ckpt',max_checkpoints=2)
model.fit({'input': X}, {'target': y_train}, n_epoch=5,
           validation_set=({'input': testX}, {'target': y_test}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
# model.save("tflearncnn.model")
# model.load("tflearncnn.model")
# for i in range(100):
# 	results= model.predict([testX[i]])
# 	print(np.argmax(results))
# 	print(np.argmax(y_test[i]))
# 	print (np.argmax(results) == np.argmax(y_test[i]))

