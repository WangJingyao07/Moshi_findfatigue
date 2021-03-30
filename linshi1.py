__author__ = 'WJY: https://kaggle.com'
import tensorflow as tf

from layers import Dense, Conv2D, Flatten, Conv2DBatchNorm, AvgPool, Dropout, Activation, MaxPool, Fully_connected,Identity_block,Convolutional_block

def vgg_bn():
    return [
	#1
        Conv2D([7, 7], 64, [1, 1, 1, 1],padding='SAME'),
	Conv2DBatchNorm(64),
	Activation(tf.nn.relu),
	MaxPool([1,2,2,1],[1,1,1,1]),
	
	#2
	Convolutional_block(f = 3, filters = [64,64,256],s = 1),
	Identity_block(f = 3, filters=[64,64,256]),
	Identity_block(f = 3, filters=[64,64,256]),
	#3
	Convolutional_block(f = 3, filters = [128,128,512],s = 2),
	Identity_block(f = 3, filters=[128,128,512]),
	Identity_block(f = 3, filters=[128,128,512]),

	#4
	Convolutional_block(f = 3, filters = [256,256,1024],s = 2),
	Identity_block(f = 3, filters=[256,256,1024]),
	Identity_block(f = 3, filters=[256,256,1024]),
	Identity_block(f = 3, filters=[256,256,1024]),
	Identity_block(f = 3, filters=[256,256,1024]),
	Identity_block(f = 3, filters=[256,256,1024]),
	#5
	Convolutional_block(f = 3, filters = [512,512,2048],s = 2),
	Identity_block(f = 3, filters=[256,256,2048]),
	Identity_block(f = 3, filters=[256,256,2048]),
	#
	AvgPool([1,2,2,1],[1,1,1,1]),
        Flatten(),
        #Dense(128),
        #Activation(tf.sigmoid),

        #Dropout(0.5),

        Dense(10),
	#Fully_connected(),
        Activation(tf.nn.softmax),
    ]

import tensorflow as tf

from layers import Dense, Conv2D, Flatten, Conv2DBatchNorm, AvgPool, Dropout, Activation

def vgg_bn():
    return [
        Conv2D([10, 10], 32, [1, 5, 5, 1]),
        Conv2DBatchNorm(32),
	Dropout(0.5),
        Activation(tf.nn.relu),

        Conv2D([3, 3], 32, [1, 1, 1, 1]),
	Dropout(0.5),
        Conv2DBatchNorm(32),
        Activation(tf.nn.relu),

        Conv2D([3, 3], 64, [1, 2, 2, 1]),
	Dropout(0.5),
        Conv2DBatchNorm(64),
        Activation(tf.nn.relu),

        Conv2D([3, 3], 64, [1, 1, 1, 1]),
	Dropout(0.5),
        Conv2DBatchNorm(64),
        Activation(tf.nn.relu),

        Conv2D([3, 3], 128, [1, 2, 2, 1]),
	Dropout(0.5),
        Conv2DBatchNorm(128),
        Activation(tf.nn.relu),

        Conv2D([2, 2], 128, [1, 1, 1, 1]),
        Conv2DBatchNorm(128),
        Activation(tf.nn.relu),

        Flatten(),

        Dense(128),
        Activation(tf.sigmoid),

        Dropout(0.5),

        Dense(10),
        Activation(tf.nn.softmax),
    ]






