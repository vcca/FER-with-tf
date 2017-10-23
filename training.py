#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:43:08 2017

@author: vcca
"""
import tensorflow as tf
import image_input
import list_model
import numpy as np
import os
import math
from datetime import datetime
from time import time
import tool
from sklearn.svm import SVC
#from sklearn.model_selection import GridSearchCV


#%%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
keep_prob = tf.placeholder("float")
N_CLASSES = 7
IMG_W = 48  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 48   
BATCH_SIZE = 32
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.01 # with current parameters, it is suggested to use learning rate<0.0001
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 1])
y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
log_dir = 'D:\Leung\lenet5\log\log1017_cnn_c0.3'
#log_dir = 'D:\Leung\lenet5\log\log0915_raf'
#file_dir='D:\Leung\lenet5\data\list_patition_label.txt'
file_dir='D:\Leung\RAF-DB\list_patition_label.txt'


#%%
def train():
#    train_dir='/media/vcca/新梗结衣/extendck48_chose'
#    train_dir='D:\Leung\lenet5\data\ck48_train'
   # tes_dir='/media/vcca/新梗结衣/emotion/ck48_tes'
    is_training = tf.placeholder(tf.bool)
    train, train_label,val, val_label= image_input.getfrom_raf(file_dir, True,ratio=0.1)

#    train, train_label, val, val_label = image_input.get_file(train_dir,True,0.1)
    train_batch, train_label_batch = image_input.get_batch(train,
                                                  train_label,
                                                  BATCH_SIZE,
                                                  True)
    
    val_batch, val_label_batch = image_input.get_batch(val,
                                                  val_label,
                                                  BATCH_SIZE,
                                                  False)
    #pre_logits output nodes 1024
    pre_logits = list_model.inference(x, N_CLASSES, is_train=is_training,keep_prob=keep_prob)
    #center_loss and center_op
    center_loss, centers_update_op = list_model.center_loss(pre_logits,y_,0.5,7)
    
    logits = tool.FC_Layer('fc7',pre_logits,out_nodes=7)
    loss = list_model.losses(logits, y_)
    total_loss = loss + 0.05 * center_loss
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([centers_update_op]):
        train_op ,lr_rate= list_model.training(loss, learning_rate) 
    acc = list_model.evaluation(logits, y_)
    
#    with tf.Session() as sess:        
    with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as sess:
#        with tf.device("/gpu:0"):
            #store variable in the batch_norm
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

        # saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        summary_op = tf.summary.merge_all()        
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
#        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
#                    input pipleline
#                _, tra_loss,tra_acc = sess.run([train_op, loss, acc])
                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _,_ops,tra_loss, tra_acc, summary_str ,lr= sess.run([train_op, update_ops , total_loss, acc, summary_op,lr_rate],
                                                feed_dict={x : tra_images, y_ : tra_labels, is_training : True,keep_prob : 0.4})
                if step % 10 == 0:
                    print('%s,Step %d, train loss = %.6f, train accuracy = %.5f ,lr = %f' %(datetime.now(),step, tra_loss, tra_acc,lr))
                    # summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                    
                if step % 100 == 0 or (step + 1) == MAX_STEP:
#                        sess.run(val_batch)
#                        _, val_loss,val_acc = sess.run([train_op, loss, acc])
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([total_loss, acc], 
                                                 feed_dict={x:val_images, y_:val_labels,is_training:False,keep_prob : 1.0})
                    print('**%s, Step %d, val loss = %.6f, val accuracy = %.5f  **' %(datetime.now(),step, val_loss, val_acc))
#                    summary_str = sess.run(summary_op)
#                    val_writer.add_summary(summary_str, step)  
                                    
                if step % 1000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)
#%%
def evaluate():
#    file_dir='D:\Leung\lenet5\data\list_patition_label.txt'  
    with tf.Graph().as_default():
#        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 1])
#        y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        is_training = tf.placeholder(tf.bool)
        keep_prob = tf.placeholder("float")
        count=0
        pred_label=[]
        true_label=[]
#        tes_dir='D:\Leung\lenet5\data\ck48_tes'
        tes_images,tes_labels = image_input.getfrom_raf(file_dir,False)
#        tes_images,tes_labels = image_input.get_file(tes_dir,False,0)
        n_test=len(tes_labels)
        tes_batch,label_batch = image_input.get_batch(tes_images,
                                                      tes_labels,
                                                      BATCH_SIZE,
                                                      False)
        pre_logits = list_model.inference(tes_batch,N_CLASSES,is_train=is_training,keep_prob=keep_prob)
        logits = tool.FC_Layer('fc7',pre_logits,out_nodes=7)
        y_pred = tf.argmax(logits,1)
        top_k_op = tf.nn.in_top_k(logits, label_batch, 1)
#        class_pred = np.zeros(shape=n_test, dtype=np.int)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt.model_checkpoint_path='D:\Leung\lenet5\log\log1017_cnn_c0.3\model.ckpt-6000'
                global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)          
            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
#                while step<1:
                while step < num_iter and not coord.should_stop():
#                    class_test=np.append(class_test,tes_label)
#                    sess.run([tes_batch, label_batch])
                    t_label,logits_label,class_pred,predictions= sess.run([label_batch,logits,y_pred,top_k_op],
                                                                        feed_dict={is_training:False,keep_prob:1.0})
#                    logits_label,class_pred = sess.run([logits, y_pred] )
                    pred_label=np.append(pred_label,class_pred)
                    true_label=np.append(true_label,t_label)
                    true_count += np.sum(predictions)
                    step += 1
                    precision = float(true_count) / total_sample_count
#                    print class_pred
                print('precision = %.5f' % precision)
                print(true_count, total_sample_count)
#                print(pred_label)
                for i in range(n_test):
                    if pred_label[i]==true_label[i]:
                        count+=1
                print(count)
                list_model.plot_confusion_matrix(pred_label,true_label)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
#%%      
def svm_evaluate():
    file_dir='D:\Leung\RAF-DB\list_patition_label.txt'
    
    with tf.Graph().as_default():
        BATCH_SIZE = 32
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 1])
        y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        is_training = tf.placeholder(tf.bool)
        keep_prob = tf.placeholder("float")
        x_train=np.zeros((1,2048))
        y_train=[]
        x_test=np.zeros((1,2048))
        y_test=[]
#        pred_label=[]
#        true_label=[]
#        tes_dir='D:\Leung\lenet5\data\ck48_tes'
        train, train_label,val, val_label= image_input.getfrom_raf(file_dir,True)
        tes_images,tes_labels = image_input.getfrom_raf(file_dir,False)
#        tes_images,tes_labels = image_input.get_file(tes_dir,False,0)
        n_test=len(tes_labels)
        n_train = len(train_label)
        tes_batch,label_batch = image_input.get_batch(tes_images,
                                                      tes_labels,
                                                      BATCH_SIZE,
                                                      False)
        tra_batch,tra_label_batch = image_input.get_batch(train,
                                                          train_label,
                                                          BATCH_SIZE,
                                                          False)
        pre_logits = list_model.inference(x,N_CLASSES,is_train=is_training,keep_prob=keep_prob)
        saver = tf.train.Saver(tf.global_variables())
        with tf.device("/gpu:0"):
            with tf.Session() as sess:
                print("Reading checkpoints...")
                ckpt = tf.train.get_checkpoint_state(log_dir)
                if ckpt and ckpt.model_checkpoint_path:
#                    ckpt.model_checkpoint_path='D:\Leung\lenet5\log\log0928_alone_conv4\model.ckpt-11999'
                    global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')
                    return
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess = sess, coord = coord)
                
                try:
                    tra_num_iter = int(math.ceil(n_train / BATCH_SIZE))
                    tes_num_iter = int(math.ceil(n_test / BATCH_SIZE))
    #                true_count = 0
    #                total_sample_count = num_iter * BATCH_SIZE
                    step = 0
                    t0 = time()
    #                while step<1:
                    print("Start extract tra_features!")
                    while step < tra_num_iter and not coord.should_stop():
    
                        tra_image,tra_label = sess.run([tra_batch, tra_label_batch])
                        feature = sess.run([pre_logits],
                                           feed_dict={x:tra_image,y_:tra_label,is_training:False,keep_prob:0.4})
                        feature = np.array(feature)
                        feature = feature.reshape((32,2048))
#                        print('step: %s' %step)
#                        print(feature.shape)
                        step += 1
    #                    feature = np.reshape(feature,[1,2048])
                        x_train = np.vstack((x_train,feature))
                        y_train = np.append(y_train,tra_label)
    #                x_train=np.reshape()
                    print("done in %0.3fs" % (time() - t0))
                    print(x_train.shape,y_train.shape)
                    
                    print("Start extract tes_features!")
                    t0 = time()
                    step = 0
                    while step < tes_num_iter and not coord.should_stop():
                        
                        tes_image,tes_label = sess.run([tes_batch,label_batch])
                        feature = sess.run([pre_logits],
                                           feed_dict={x:tes_image,y_:tes_label,is_training:False,keep_prob:1.0})
                        feature = np.array(feature)
                        feature = feature.reshape((32,2048))
#                        print('step: %s' %step)
#                        print(feature.shape)
                        
                        x_test = np.vstack((x_test,feature))
                        y_test = np.append(y_test,tes_label)
                        step += 1
                    print("done in %0.3fs" % (time() - t0))    
                    print(x_test.shape,y_test.shape)
    #                list_model.plot_confusion_matrix(pred_label,true_label)
                    
                except Exception as e:
                    coord.request_stop(e)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    
                print("Fitting the classifier to the training set")
                t0 = time()
#                param_grid = {'C': [1e3, 1e5],
#                  'gamma': [0.005, 0.01], }
                clf = SVC(C=1.0,kernel='rbf',gamma=0.012,cache_size=2000,class_weight='balanced',decision_function_shape='ovr')
#                clf = GridSearchCV(SVC(kernel='rbf',cache_size=2000, class_weight='balanced'), param_grid)
                clf = clf.fit(x_train[1:-17,], y_train[:-17])
                
                print("fit classfier done in %0.3fs" % (time() - t0))
                print(clf)
#                print("Best estimator found by grid search:")
#                print(clf.best_estimator_)
                
                t0 = time()
                y_pred = clf.predict(x_test[1:-4,])
                print("done in %0.3fs" % (time() - t0))
                
                list_model.plot_confusion_matrix(y_pred,y_test[:-4])
            
            
            
             
#%%
if __name__ == '__main__': 
    train()
    evaluate()
#    svm_evaluate()
         