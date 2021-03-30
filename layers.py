######################################
__author__ = 'WJY: https://kaggle.com'
########################################
import math
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from keras import layers
from utilities import weight_bias

class Dropout:
    def __init__(self, keep_prob, name='dropout'):
        self.keep_prob = keep_prob
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            keep_prob = tf.where(model.is_training, self.keep_prob, 1.0)
            self.h = tf.nn.dropout(x, keep_prob)
            return self.h

class Dense:
    def __init__(self, fan_out, name='dense'):
        self.fan_out = fan_out
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            input_shape = x.get_shape()
            fan_in = input_shape[-1].value
            stddev = math.sqrt(1.0 / fan_in) # he init

            shape = [fan_in, self.fan_out]
            W, b = weight_bias(shape, stddev=stddev, bias_init=0.0)

            self.h = tf.matmul(x, W) + b
            return self.h

class Activation:
    def __init__(self, activation, name='activation'):
        self.name = name
        self.activation = activation

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            self.h = self.activation(x)
            return self.h

class MaxPool:
    def __init__(self, ksize, strides, padding='VALID', name='max_pool'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            self.h = tf.nn.max_pool(x, self.ksize, self.strides, self.padding)
            return self.h

class GlobalAvgPool:
    def __init__(self, name='global_avg_pool'):
        self.name = name

    def apply(self, x, index, model):
        input_shape = x.get_shape().as_list()
        k_w, k_h = input_shape[1], input_shape[2]
        with tf.name_scope(self.name):
            self.h = tf.nn.avg_pool(x, [1, k_w, k_h, 1], [1, 1, 1, 1], 'VALID')
            return self.h

class AvgPool:
    def __init__(self, ksize, strides, padding='VALID', name='avg_pool'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            self.h = tf.nn.avg_pool(x, self.ksize, self.strides, self.padding)
            return self.h

class Input:
    def __init__(self, input_placeholder):
        self.h = input_placeholder

    def apply(self, x, index, model):
        return self.h

class Conv2D:
    def __init__(self, filter_shape, output_channels, strides, padding='VALID', name='conv2d'):
        self.filter_shape = filter_shape
        self.output_channels = output_channels
        self.strides = strides
        self.padding = padding
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            input_shape = x.get_shape()
            input_channels = input_shape[-1].value

            k_w, k_h = self.filter_shape
            stddev = math.sqrt(2.0 / ((k_w * k_h) * input_channels)) # he init

            shape = self.filter_shape + [input_channels, self.output_channels]
            W, b = weight_bias(shape, stddev=stddev, bias_init=0.0)

            self.h = tf.nn.conv2d(x, W, self.strides, self.padding) + b
            return self.h

class Flatten:
    def __init__(self, name='flatten'):
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            shape = x.get_shape()
            dim = shape[1] * shape[2] * shape[3]
            self.h = tf.reshape(x, [-1, dim.value])
            return self.h

class Fully_connected:
    def __init__(self, name='fully_connected'):
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
	    self.h = tf.contrib.layers.fully_connected(x,10,activation_fn=None) 
	    return self.h

class Convolutional_block:
    def __init__(self, f, filters, s, name='convolutional_block'):
	self.name = name
	self.f = f
	self.filters = filters
	self.s = s

    def apply(self,x,index,model):
	with tf.name_scope(self.name):
	    F1,F2,F3 = self.filters
	    f = self.f
	    s = self.s
	    x_shortcut = x
	    #1
	    z1 = Conv2D([1, 1], F1, [1, s, s, 1])
	    z1 = z1.apply(x,0,model)
	    Dropout(0.5)
	    c1 = Conv2DBatchNorm(F1)
	    z1 = c1.apply(z1,0,model)
	    
	    a1 = Activation(tf.nn.relu)
	    a1 = a1.apply(z1,0,model)
	    #2
	    z2 = Conv2D([f, f], F2, [1, 1, 1, 1],padding='SAME')
	    z2 = z2.apply(a1,1,model)
	    Dropout(0.5)
	    c2 = Conv2DBatchNorm(F2)
	    z2 = c1.apply(z2,1,model)
	    Dropout(0.5)
	    a2 = Activation(tf.nn.relu)
	    a2 = a2.apply(z2,1,model)
	    #3
	    z3 = Conv2D([1, 1], F3, [1, 1, 1, 1])
	    z3 = z3.apply(a2,2,model)
	    Dropout(0.5)
	    c3 = Conv2DBatchNorm(F3)
	    z3 = c3.apply(z3,2,model)
	    #residual networks
	    
	    r4 = Conv2D([1, 1], F3, [1, s, s, 1])	    
	    r4 = r4.apply(x_shortcut,3,model)
	    Dropout(0.5)
	    rb4 = Conv2DBatchNorm(F3)
	    rb4 = rb4.apply(r4,2,model)
	    
	    X = layers.add([z3, rb4])
	    a3 = Activation(tf.nn.relu)
	    a3 = a3.apply(X,2,model)
	    self.h = a3
	    return self.h

class Identity_block:
    def __init__(self, f, filters, name='identify_block'):
	self.name = name
	self.f = f
	self.filters = filters

    def apply(self,x,index,model):
	with tf.name_scope(self.name):
	    F1,F2,F3 = self.filters
	    f = self.f
	    x_shortcut = x
	    #1
	    z1 = Conv2D([1, 1], F1, [1, 1, 1, 1])
	    z1 = z1.apply(x,0,model)
	    Dropout(0.5)
	    c1 = Conv2DBatchNorm(F1)
	    z1 = c1.apply(z1,0,model)
	    
	    a1 = Activation(tf.nn.relu)
	    a1 = a1.apply(z1,0,model)
	    #2
	    z2 = Conv2D([f, f], F2, [1, 1, 1, 1],padding='SAME')
	    z2 = z2.apply(a1,1,model)
	    Dropout(0.5)
	    c2 = Conv2DBatchNorm(F2)
	    z2 = c1.apply(z2,1,model)
	    Dropout(0.5)
	    a2 = Activation(tf.nn.relu)
	    a2 = a2.apply(z2,1,model)
	    #3
	    z3 = Conv2D([1, 1], F3, [1, 1, 1, 1])
	    z3 = z3.apply(a2,2,model)
	    Dropout(0.5)
	    c3 = Conv2DBatchNorm(F3)
	    z3 = c3.apply(z3,2,model)
	    #residual networks
	    X = layers.add([z3, x_shortcut])
	    a3 = Activation(tf.nn.relu)
	    a3 = a3.apply(X,3,model)
	    self.h = a3
	    return self.h
class Conv2DBatchNorm:

    def __init__(self, fan_out, affine=True, name='batch_norm'):
        self.fan_out = fan_out
        self.affine = affine
        self.name = name

    def apply(self, x, index, model):
        with tf.name_scope(self.name):
            beta = tf.Variable(tf.constant(0.0, shape=[self.fan_out]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[self.fan_out]), name='gamma', trainable=self.affine)

            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

            def mean_var_with_update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = control_flow_ops.cond(model.is_training, mean_var_with_update, lambda: (ema_mean, ema_var))

            self.h = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, self.affine)
            return self.h
