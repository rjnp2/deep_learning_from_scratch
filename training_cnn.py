#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:45:46 2019

@author: rjn
"""

#importing all neccessary packages
import numpy as np
from deeplearning.Layers import Conv2D,Dense,MaxPooling2D,Dropout
from deeplearning.Layers import Flatten, NeuralNetwork, Activation
import numpy as np
from deeplearning.activat import ReLU
from deeplearning.utils import train_test_split
from deeplearning.loss import SquareLoss
from deeplearning.optimizer import RMSprop
import pickle 

class createCNNModel:
    '''
    to create cnn model
    using built-in functions of keras such as
    conv2D,
    maxpooling2D,
    flatten,
    dropout
    '''

    def __init__(self,input_size, output_size, epochs, batch_size):
        
        self.input_size = input_shape
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.createmodel()


    def createmodel(self):

        #creating a Neural network is to initialise the network using the NeuralNetwork Class from deeplearning.
        
        clf = NeuralNetwork(optimizer=RMSprop() , loss= SquareLoss)
        
        # The first two layers with 64 filters of window size 3x3
        # filters : Denotes the number of Feature detectors
        # kernel_size : Denotes the shape of the feature detector. (3,3) denotes a 3 x 3 matrix.
        # input _shape : standardises the size of the input image
        # activation : Activation function to break the linearity
        clf.add(Conv2D(n_filters=32, filter_shape=(3,3), stride=1, input_shape=self.input_size,
                       biase=False, padding='valid'))
        clf.add(Activation(ReLU))
        clf.add(Conv2D(n_filters=32, filter_shape=(3,3), biase=False, stride=1, padding='same'))
        clf.add(Activation(ReLU))
        
        # pool_size : the shape of the pooling window.
        clf.add(MaxPooling2D(pool_shape=(2, 2), stride=2))
        clf.add(Dropout(0.2))
    
        clf.add(Conv2D(n_filters=64, filter_shape=(3,3),  biase=False, stride=1, padding='same' ))
        clf.add(Activation(ReLU))
        clf.add(Conv2D(n_filters=64, filter_shape=(3,3),  biase=False, stride=1, padding='same' ))
        clf.add(Activation(ReLU))
        clf.add(MaxPooling2D(pool_shape=(2, 2), stride=2))
        
        clf.add(Conv2D(n_filters=128, filter_shape=(3,3), stride=1, padding='same'))
        clf.add(Activation(ReLU))
        clf.add(Conv2D(n_filters=128, filter_shape=(3,3), stride=1, padding='same'))
        clf.add(Activation(ReLU))
        clf.add(MaxPooling2D(pool_shape=(2, 2), stride=2))
        
        clf.add(Dropout(0.2))
        clf.add(Conv2D(n_filters=256, filter_shape=(3,3), stride=1, padding='same'))
        clf.add(Activation(ReLU))
        clf.add(Conv2D(n_filters=256, filter_shape=(3,3), stride=1, padding='same'))
        clf.add(Activation(ReLU))
        clf.add(MaxPooling2D(pool_shape=(2, 2), stride=2))

    
        clf.add(Flatten())

        # units: Number of nodes in the layer.
        # activation : the activation function in each node.
        clf.add(Dropout(0.2))
        clf.add(Dense(512))
        clf.add(Activation(ReLU))
        clf.add(Dropout(0.2))
        
        clf.add(Dense(self.output_size))
        
        self.clf = clf
        
        return self.clf
                
    def summary(self , name="ConvNet"):
        
        return self.clf.summary(name="ConvNet")
    
    def compile(self,X_train, X_test, y_train, y_test,save= True):
        
        self.hist = self.clf.fit(X=X_train, y= y_train , n_epochs=self.epochs,
                                 batch_size= self.epochs, val_set=(X_test,y_test))

         
    def save_history(self):
        
        # for visualizing losses and accuracy
        # History.history attribute is a record of training loss values
        # metrics values at successive epochs 
        # as well as validation loss values and validation metrics values
        train_loss= self.hist[0]
        val_loss= self.hist[1]
        xc=range(self.epochs)
        
        model_histroy = [
                train_loss,
                val_loss,                
                xc]
        
        np.save('model_histroy.npy',model_histroy)

#%%

#loading train_data and test_data 
train_data = np.load('/home/rjn/Pictures/major project/data/train_name.npy')
land_data = np.load('/home/rjn/Pictures/major project/data/land_name.npy')

print('original_shape: \t' , train_data.shape)
print(land_data.shape)

#pre-processing all data   
train_data = np.array(train_data).reshape(-1, 96,96,1)

print(train_data.shape)
#%%

#saving rows,col,dim of datasets to input_shape variable
#to give input size /shape to CNN model 
input_shape = (None , ) + train_data.shape[1:]
output_size = land_data.shape[1:][0]

#Split arrays or matrices into random train and test subsets
train_data ,vtrain_data ,land_data, vland_data = train_test_split(train_data, land_data  ,
                                                       test_size=0.08, seed=32)
print(train_data.shape)
print(land_data.shape)

print(vtrain_data.shape)
print(vland_data.shape)

#%%
epochs = 5
batch_size = 32

model = createCNNModel(input_shape , output_size ,epochs, batch_size )

model.summary()
model.compile(train_data,  vtrain_data , land_data, vland_data)
model.save_history()
