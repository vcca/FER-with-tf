#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:37:32 2017

@author: vcca
"""
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import tool
#%%
class_names=['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']
def inference(x,is_train,keep_prob):

    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    x = tool.conv('conv1', x, 64, scale=0.0002,kernel_size=[3,3],stride=[1,1,1,1],is_train=is_train)
    #pool1
    x = tool.pool('pool1', x, kernel=[1,2,2,1], strides=[1,2,2,1])
    #conv2
    x = tool.conv('conv2', x, 96, scale=0.0002,kernel_size=[3,3],stride=[1,1,1,1],is_train=is_train)
    #pool2
    x = tool.pool('pool2', x, kernel=[1,2,2,1], strides=[1,2,2,1])
    #conv3
    x = tool.conv('conv3', x, 256, scale=0.0002, kernel_size=[3,3],stride=[1,1,1,1],is_train=is_train)
    #conv4
#    x = tool.conv('conv4', x, 128, scale=0.0001, kernel_size=[3,3],stride=[1,1,1,1],is_train=is_train)
    #pool4
    x = tool.pool('pool4', x, kernel=[1,2,2,1], strides=[1,2,2,1])
    #conv5
    x = tool.conv('conv5', x, 256, scale=0.0002,kernel_size=[3,3],stride=[1,1,1,1],is_train=is_train)
#    conv6
#    x = tool.conv('conv6', x, 256, scale=0.0003, kernel_size=[3,3],stride=[1,1,1,1],is_train=is_train)
    #fc
    # x = tf.nn.dropout(x, keep_prob=0.5)
    
    x = tool.FC_Layer('fc6', x,scale=0.0002,out_nodes=1024)

    x = tf.layers.batch_normalization(x,training=is_train)

    x = tf.nn.dropout(x, keep_prob = keep_prob)

    x = tf.nn.relu(x)

    #fc2
#    x = tool.FC_Layer('fc7', x, out_nodes=n_classes)
    
    return x
      
#%%
def losses(logits, labels,alpha=1,gamma=0):

    with tf.variable_scope('loss') as scope:
        L2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#        use one-hot encoder
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')        
#        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
#                        (logits=logits, labels=labels, name='xentropy_per_example')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='loss')
#        epsilon=1e-7        
#        preds = tf.cast(tf.argmax(logits,1),tf.int32)
#        pt = tf.where(tf.equal(labels, preds),logits, 1-logits)
#        focal_loss = tf.reduce_mean(- alpha * tf.pow(1.-pt, gamma) * tf.log(tf.cast(labels,tf.float32)), name='focal_loss')
#        loss = L2_loss + focal_loss
        loss = L2_loss + cross_entropy_loss
        tf.summary.scalar(scope.name+'/L2_loss', L2_loss)
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def center_loss(features, labels, alpha, num_classes):
    len_features = features.get_shape()[1]
    
    labels = tf.argmax(labels,1)
    centers = tf.get_variable('centers',[num_classes,len_features],dtype=tf.float32,
                              initializer=tf.constant_initializer(0),trainable=False)
    labels = tf.reshape(labels,[-1])
    
    centers_batch = tf.gather(centers, labels)
    weight = features-centers_batch
#    print(labels.get_shape()[0])
#    for i in range(labels.get_shape()[0]):
#        if(labels[i]==1):
#            weight[i] = 5 * weight[i]
#        if(labels[i]==2 or labels[i]==5):
#            weight[i] = 3 * weight[i]
#        if(labels[i]==3):
#            weight[i] = 0.5 * weight[i]
    diff = centers_batch - features
    
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count,unique_idx)
    appear_times = tf.reshape(appear_times,[-1,1])
    
    diff = diff / tf.cast((1+appear_times),tf.float32)
    diff = alpha * diff
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    loss = tf.div(tf.nn.l2_loss(weight),int(len_features))
    tf.summary.scalar('center_loss', loss)
    return loss, centers_update_op


#%%
def training(loss, str_learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(str_learning_rate,
                                           global_step=global_step,
                                           decay_steps=1000,
                                           decay_rate=0.5)
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
      
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op,learning_rate

#%%
def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
      
      labels = tf.argmax(labels,1) #      one-hot encoder
      correct = tf.nn.in_top_k(logits, labels, k=1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

#%%
def plot_confusion_matrix(class_pred,class_test):
    cm = confusion_matrix(y_true=class_test,
                          y_pred=class_pred)
    acc=0
    for i in range(7):
        class_name = "{}".format(class_names[i])
        print(cm[i,:],class_name)
        acc = acc + cm[i,i]/float(cm[i:i+1].sum())
#        num+=cm[i,i]
    print(acc/7.0)
        
    class_numbers = [" ({0})".format(i) for i in range(7)]
    print("".join(class_numbers))