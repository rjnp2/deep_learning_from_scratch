#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:13:41 2020

@author: rjn
"""

from deeplearning.utils import batch_iterator
from tabulate import tabulate
import sys
from tqdm import tqdm
import time
import datetime
import pickle
import cupy as cp
import numpy as np

class NeuralNetwork():
    
    """Neural Network. Deep Learning base model.

    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance.
    """
    
    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()

    def set_trainable(self, trainable):
        """ Method which enables freezing of the weights of the network's layers."""
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """ Method which adds a layer to the neural network """
        # If this is not the first layer added then set the input shape
        # to the output shape of the last added layer
        
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].determin_output_shape())
            layer.valid_layer()

        # If the layer has weights that needs to be initialized 
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
            
        # Add layer to the network
        self.layers.append(layer)

    def test_on_batch(self, X, y):
        """ Evaluates the model over a single batch of samples """
        y_pred = self._forward_pass(cp.asarray(X), training=False)
        loss = cp.mean(self.loss_function.loss(cp.asarray(y), y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc

    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._forward_pass(cp.asarray(X))
        cp.cuda.Stream.null.synchronize()
        loss = self.loss_function.loss(cp.asarray(y), y_pred)
        cp.cuda.Stream.null.synchronize()
      
        # acc = self.loss_function.acc(y, y_pred)
        
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(cp.asarray(y), y_pred)
        cp.cuda.Stream.null.synchronize()

        # # Backpropagate. Update weights
        self._backward_pass(loss_grad=loss_grad)
        cp.cuda.Stream.null.synchronize()

        return loss

    def fit(self, X, y, n_epochs, batch_size, val_set=None):
        """ Trains the model for a fixed number of epochs """
        for epo in range(1, n_epochs+1):
            
            print('n_epochs: ', epo ,end ='\n')
            batch_error = []
            start = time.time()
            tot = np.round(X.shape[0] / batch_size)
            i = 0
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                
                X_batch = X_batch.astype('float32')
                X_batch = X_batch / 255

                y_batch = y_batch.astype('float32')
                y_batch = (y_batch - 48) / 48
                loss = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
                t = datetime.timedelta(seconds= (time.time() - start)) 
                i += 1
                sys.stdout.write('\r' + 'time:  ' + str(t) + '  complete: ' +  
                                  str(np.round((i/tot),3))  + '  t_loss: ' + str(loss))

            self.errors["training"].append(np.mean(batch_error))
            print('\t')
            if val_set is not None:

                for X_batch, y_batch in batch_iterator(val_set[0], val_set[1], batch_size=batch_size):
                
                    X_batch = X_batch.astype('float32')
                    X_batch = X_batch / 255

                    y_batch = y_batch.astype('float32')
                    y_batch = (y_batch - 48) / 48

                    val_loss, _ = self.test_on_batch(X_batch, y_batch)
                    sys.stdout.write('\r' + '  val_loss:', str(val_loss))
                self.errors["validation"].append(val_loss)
             
            if save_files:
              self.save()  
      
            if callback:
              callback(self)              
            print()
   
        del X,y, X_batch , y_batch , val_set
            
        return self.errors["training"], self.errors["validation"]

    def save(self , name = 'model.pkl'):

        name = '/home/rjn/Pictures/' + name
        pickle.dump(self, open(name, 'wb'),pickle.HIGHEST_PROTOCOL)

    def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN """
        
        for layer in self.layers:
            X = layer.forward_pass(X.copy(), training)
            
        return X

    def _backward_pass(self, loss_grad):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)
                
    def get_weights(self):
        
        table_data = np.array([['layer_name', 'parameterW', 'parameterb']])
        
        for n,layer in tqdm(enumerate(self.layers)):
            
            parameter = layer.load_parameters()
            layer_name = layer.layer_name() + '_' + str(n)
            
            W = parameter['W']   
            b = parameter['b']
            table_data = np.append( table_data, [[layer_name,W,b]],axis =0)
            
        print()
        print(tabulate(table_data[:,:1],tablefmt="fancy_grid"))
                
        return table_data
    
    def load_weights(self,loader):
        
        for n,values in enumerate(zip(self.layers, loader)):
            
            layer = values[0]
            
            values = values[1]
            
            print(values[0])
            
            shap = layer.load_parameters()
                            
            if shap is not None:
            
                shap = (cp.asarray(shap["W"]).shape , cp.asarray(shap["b"]).shape)     
                print('orig ' , shap)
                W = cp.asarray(values[1])
                
                b = cp.asarray([])
                
                if values[2] is not None:
                    b = cp.asarray(values[2])
                
                sshap = (W.shape, b.shape)  
                
                print('loader ' , sshap)
                
                if shap == sshap :
                    
                    layer.set_weights((W,b))  
                    
                    shap = layer.load_parameters()
                    
                    shap = (shap["W"].shape , shap["b"].shape) 
                    print('after ' , sshap)
            print()
              
    def summary(self, name="Model Summary"):
        # Print model name
        # Network input shape (first layer's input shape)

        # Iterate through network and get each layer's configuration
        table_data = [ ["Input Shape:", self.layers[0].input_shape],
                      ["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0

        for n,layer in tqdm(enumerate(self.layers)):
            layer_name = layer.layer_name() + '_' + str(n)
            params = layer.parameters()
            out_shape = layer.determin_output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params
        # Print network configuration table
        table_data.append(["Total Parameters: ", tot_params ])
        print()
        print(tabulate(table_data,tablefmt="grid"))
        
        del table_data

    def predict(self, X):
        """ Use the trained model to predict labels of X """
        X = cp.asarray(X) / 255
        return self._forward_pass(X).get()
