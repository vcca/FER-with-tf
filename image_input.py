# -*- coding: utf-8 -*-
"""
vccaLeung Editor
"""

import tensorflow as tf
import numpy as np
import os
import math

def getfrom_raf(file,is_train,ratio=0):
    images = []
    labels = []
    temp = []
    for line in open(file):
        if is_train:
            if(line.split('\\')[-1][0:5]=='train'):
                images.append(line.split(' ')[0])
                labels = np.append(labels, int(line.split(' ')[1][-2]))
        else:
            #test_data
            if(line.split('\\')[-1][0:4]=='test'):
                images.append(line.split(' ')[0])
                labels = np.append(labels, int(line.split(' ')[1][-2]))
    temp=np.array([images,labels])
    temp=temp.transpose()
    np.random.shuffle(temp)
    
    images_list = temp[:, 0]
    label_list = temp[:, 1]
    n_sample = len(label_list)
    if is_train:
        n_val = math.ceil(n_sample*ratio)
        n_train = n_sample-int(n_val)        
        tra_images = images_list[0:n_train]
        tra_labels = label_list[0:n_train]
        tra_labels = [int(float(i)-1) for i in tra_labels]
        val_images = images_list[n_train:-1]
        val_labels = label_list[n_train:-1]
        val_labels = [int(float(i)-1) for i in val_labels]
        
        return tra_images,tra_labels,val_images,val_labels
    else:
        labels_list = [int(float(i)-1) for i in label_list]
        
    return images_list,labels_list
                                                              
                
def get_file(file_dir,is_train,ratio):
	
    images = []
    temp = []
    labels = []
    for root, sub_folders, files in os.walk(file_dir):
        for filename in files:
            filepath=os.path.join(root,filename)
            images.append(filepath)
            labels=np.append(labels,filepath.split('\\')[-2])
    temp=np.array([images,labels])
    temp=temp.transpose()
    np.random.shuffle(temp)

#    image_list = temp[:, 0]
#    label_list = temp[:, 1]
#    n_sample = len(label_list)
    if is_train:
        np.random.shuffle(temp)

        image_list = temp[:, 0]
        label_list = temp[:, 1]
        n_sample = len(label_list)
        n_val = math.ceil(n_sample*ratio)
        n_train = n_sample-int(n_val)
        
        tra_images = image_list[0:n_train]
        tra_labels = label_list[0:n_train]
        tra_labels = [int(float(i)-1) for i in tra_labels]
        val_images = image_list[n_train:-1]
        val_labels = label_list[n_train:-1]
        val_labels = [int(float(i)-1) for i in val_labels]
    #    label_list = [int(float(i)) for i in label_list]
                 
        return tra_images,tra_labels,val_images,val_labels
    else:
        
        image_list = temp[:, 0]
        label_list = temp[:, 1]
        n_sample = len(label_list)
        tes_images = image_list
        tes_labels = label_list
        tes_labels = [int(float(i)-1) for i in tes_labels]
        
        return tes_images,tes_labels


def get_batch(image,label,batch_size,is_train):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
#    if is_train:
    input_queue = tf.train.slice_input_producer([image, label])
        
#    else:
#        input_queue = tf.train.slice_input_producer([image, label],shuffle=False)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents,channels=1)
    image = tf.image.resize_image_with_crop_or_pad(image, 48, 48)
    if is_train:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
#        image = tf.image.random_contrast(image,lower=0.9,upper=1.1)
#        image = tf.contrib.keras.preprocessing.image.random_rotation(image,20)
#        image = tf_image.rotate(image,
#                                tf.random_uniform((), minval=-np.pi/12, maxval=np.pi/12))
    image = tf.image.per_image_standardization(image)
    
    if is_train:
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                    batch_size= batch_size,
                                                    num_threads= 64, 
                                                    capacity = 2000,
                                                    min_after_dequeue=1500)
    else:
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size= batch_size,
                                                  num_threads= 64, 
                                                  capacity = 2000)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch