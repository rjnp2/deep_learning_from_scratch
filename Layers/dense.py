#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:49:26 2020

@author: rjn
"""

import copy
from . import Layer
import cupy as cp
import numpy as np

class Dense(Layer):
    
    """A fully-connected NN layer.
    
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """
    
    def __init__(self, n_units : int,
                 input_shape: tuple =None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
                
    def valid_layer(self):
        
        self.output_shape = self.determin_output_shape()
        
        assert self.n_units > 0 , 'hidden should be greater than 0'

    def initialize(self, optimizer):
        # Initialize the weights
        self.W  = cp.random.normal(loc=0.0, scale = np.sqrt(2/(self.input_shape[1] + self.n_units)), 
                                        size = ( self.input_shape[1],self.n_units))
        self.w0 = cp.random.normal(loc=0.0, scale = np.sqrt(2/self.n_units), size = (self.n_units) )
        # Weight optimizers
        self.opt  = copy.copy(optimizer)

    def parameters(self):
        
        '''
        
        Returns parameters
       
        '''
        return np.prod(self.W.shape) + np.prod(self.w0.shape)
    
    def set_weights(self, weight):
        self.W = weight[0]
        self.w0 = weight[1]
        
    def load_parameters(self):
        
        para = {'W' : self.W,
                'b': self.w0}
        return para

    def forward_pass(self, X : cp.array, training : str ):
        '''
        
        Parameters
        ----------
        X : cp.array
            array of prevoius data.
        training : str
            trainable.

        Returns
        -------
        cp.array
            output array of dense.
            
        '''

        if training :
            self.layer_input = X.copy()
        
        return cp.dot(X, self.W) + self.w0

    def backward_pass(self, accum_grad : cp.array ):
        '''
        
        Parameters
        ----------
        accum_grad : cp.array
            gradient with respect to weight.

        Returns
        -------
        cp.array
            gradient of activation.
            
        '''
        
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            n = self.layer_input.shape[0]
            # Calculate gradient w.r.t layer weights
                        
            grad_w = cp.dot(cp.asarray(self.layer_input).T, accum_grad) / n           
            grad_w0 = cp.sum(accum_grad, axis=0, keepdims=True) / n
            
            cp.cuda.Stream.null.synchronize()
            
            assert grad_w.shape == self.W.shape and grad_w0.shape == self.w0.shape
                    
            # Update the layer weights , )
            self.W , self.w0 =self.W_opt.update(w=self.W, b= self.w0 , 
                                grad_wrt_w= grad_w, grad_wrt_b =grad_w0)
            cp.cuda.Stream.null.synchronize()
            
            assert self.W.shape == W.shape
        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass

        del self.layer_input , grad_w, grad_w0

        return accum_grad.dot(W.T)

    def determin_output_shape(self):
        
        '''
        Return input shape of this object layers
        '''
        return ( None , self.n_units)
