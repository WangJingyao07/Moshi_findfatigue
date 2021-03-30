__author__ = 'WJY: https://kaggle.com'
import tensorflow as tf

from layers import Dense, Conv2D, Flatten, Conv2DBatchNorm, AvgPool, Dropout, Activation, MaxPool, Fully_connected,Identity_block,Convolutional_block

def vgg_bn():
    return [
	#1
        Conv2D([7, 7], 64, [1, 3, 3, 1]),
	Conv2DBatchNorm(64),
	Activation(tf.nn.relu),
	MaxPool([1,4,4,1],[1,1,1,1]),
	
	#2
	Convolutional_block(f = 3, filters = [64,64,256],s = 1),
	MaxPool([1,5,5,1],[1,1,1,1]),
	Dropout(0.5),
	Identity_block(f = 3, filters=[64,64,256]),
	Dropout(0.5),

	Identity_block(f = 3, filters=[64,64,256]),
	Dropout(0.5),
	MaxPool([1,2,2,1],[1,1,1,1]),
	#3
	Convolutional_block(f = 3, filters = [128,128,512],s = 2),
	Dropout(0.5),
	Identity_block(f = 3, filters=[128,128,512]),
	Dropout(0.5),
	Identity_block(f = 3, filters=[128,128,512]),
	Dropout(0.5),
	MaxPool([1,2,2,1],[1,1,1,1]),

	#4
	Convolutional_block(f = 3, filters = [256,256,1024],s = 2),
	Identity_block(f = 3, filters=[256,256,1024]),
	Identity_block(f = 3, filters=[256,256,1024]),
	Identity_block(f = 3, filters=[256,256,1024]),
	Identity_block(f = 3, filters=[256,256,1024]),
	Identity_block(f = 3, filters=[256,256,1024]),
        Flatten(),
        Dense(128),
        Activation(tf.sigmoid),

        Dropout(0.5),

        Dense(10),
	#Fully_connected(),
        Activation(tf.nn.softmax),
    ]

