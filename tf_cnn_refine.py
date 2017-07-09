# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-10 dataset classification task.
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

    CSAL4243 Introduction to Machine Learning's assignment 3.
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
## To Do. same as training.
    ## Define your network here
network = input_data(shape=[None, 32, 32, 3], 
                        data_preprocessing=img_prep, 
                        data_augmentation=img_aug) 

network = conv_2d(network, 64, 3, activation='relu') 
network = max_pool_2d(network, 2) 
network = conv_2d(network, 32, 3, activation='relu')  
network = max_pool_2d(network, 2) 
network = conv_2d(network, 32, 3, activation='relu')  
network = max_pool_2d(network, 2) 
network = fully_connected(network, 256, activation='relu') 
network = dropout(network, 0.75) 
network = fully_connected(network, 256, activation='relu') 
network = dropout(network, 0.75) 
network = fully_connected(network, 10, activation='softmax') 
network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.01) 	

# Define model object
## To Do
    ## Define model and assign network. Same as training.
model = tflearn.DNN(network, tensorboard_verbose=0) 

# Load Model into model object
## To Do.
    ## Use the model.load() function
model.load('a1.tfl')

# Refine model
# To Do.
    ## Call the fit function for retraining model
model.fit(X, Y, n_epoch=5, shuffle=True, validation_set=(X_test, Y_test),show_metric=True, batch_size=64, run_id='cifar10_cnn') 	

# Manually save model
## To Do
    ## Save model, same as training.
model.save('a2.tfl')
