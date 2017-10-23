#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 09:37:16 2017

@author: vcca
"""
import tensorflow as tf
import skimage
#%%
def conv(layer_name,x,out_channels,scale=0.0,kernel_size=[3,3],stride=[1,1,1,1], is_train=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=scale)) #L2 regularizer
        
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.layers.batch_normalization(x, training=is_train)
        x = tf.nn.relu(x, name='relu')
        split = tf.split(x,num_or_size_splits=out_channels,axis=3)
        tf.summary.image(layer_name+'/feature_map', split[0])
        return x

#%%
def pool(layer_name,x,kernel=[1,2,2,1],strides=[1,2,2,1]):
    
    x=tf.nn.max_pool(x,kernel,strides,padding='SAME',name=layer_name)
    return x

#%%
def batch_norm(x):
    
    epsilon = 1e-3
    batch_mean,batch_var = tf.nn.moments(x,[0])
    x = tf.nn.batch_normalization(x,
                                mean=batch_mean,
                                variance=batch_var,
                                offset=None,
                                scale=None,
                                variance_epsion=epsilon)
    return x


#%%
def FC_Layer(layer_name,x,out_nodes,scale=0.0,is_image=False):
    
    
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        if is_image:
#        split = tf.split(x,num_or_size_splits=,axis=3)
            tf.summary.image(layer_name+'/feature_map', x)
        return x 
    
#%%rotate image
def rotate(image,max_degree=15.0):
    uniform_random = tf.random_uniform([],minval=0,maxval=1.0)
    mirror_cond = tf.less(uniform_random, .5)
    result = tf.cond(mirror_cond,
                     lambda: skimage.transform.rotate(image,tf.random_uniform([], minval=-max_degree,maxval=max_degree)),
                     lambda: image)
    tf.convert_to_tensor(result)
    return result
                     