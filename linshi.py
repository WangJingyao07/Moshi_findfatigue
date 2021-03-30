__author__ = 'WJY: https://kaggle.com'
import tensorflow as tf

from layers import Dense, Conv2D, Flatten, Conv2DBatchNorm, AvgPool, Dropout, Activation, MaxPool

def vgg_bn():
    return [
	#1
        Conv2D([3, 3], 64, [1, 1, 1, 1], padding='SAME'),
        Conv2DBatchNorm(64),
        Activation(tf.nn.relu),
	#2
        Conv2D([3, 3], 64, [1, 1, 1, 1], padding='SAME'),
        Conv2DBatchNorm(64),
        Activation(tf.nn.relu),
	#3
	MaxPool([1,2,2,1],[1,2,2,1],padding='SAME'),

	#4
        Conv2D([3, 3], 128, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(128),
        Activation(tf.nn.relu),
	#5
        Conv2D([3, 3], 128, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(128),
        Activation(tf.nn.relu),
	#6
	MaxPool([1,2,2,1],[1,2,2,1],padding='SAME'),
	#7
        Conv2D([3, 3], 256, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(256),
        Activation(tf.nn.relu),
	#8
        Conv2D([3, 3], 256, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(256),
        Activation(tf.nn.relu),
	#9
        Conv2D([3, 3], 256, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(256),
        Activation(tf.nn.relu),
	#10
	MaxPool([1,2,2,1],[1,2,2,1],padding='SAME'),
	#11
        Conv2D([3, 3], 512, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(512),
        Activation(tf.nn.relu),
	
	#12
        Conv2D([3, 3], 512, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(512),
        Activation(tf.nn.relu),
	#13
        Conv2D([3, 3], 512, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(512),
        Activation(tf.nn.relu),
	#14
	MaxPool([1,2,2,1],[1,2,2,1],padding='SAME'),
	#15
        Conv2D([3, 3], 512, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(512),
        Activation(tf.nn.relu),
	#16
        Conv2D([3, 3], 512, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(512),
        Activation(tf.nn.relu),
	#17
        Conv2D([3, 3], 512, [1, 1, 1, 1],padding='SAME'),
        Conv2DBatchNorm(512),
        Activation(tf.nn.relu),
	#18
	MaxPool([1,2,2,1],[1,2,2,1],padding='SAME'),


        Flatten(),

        Dense(4096),
        Activation(tf.nn.relu),

        Dropout(0.5),

        Dense(4096),
        Activation(tf.nn.relu),

        Dense(1000),
        Activation(tf.nn.relu),
	Dense(10),
        Activation(tf.nn.softmax),
    ]
