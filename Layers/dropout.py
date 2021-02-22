#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:47:45 2020

@author: rjn
"""

from . import Layer
import cupy as cp

class Dropout(Layer):
    
    """
    A layer that randomly sets a fraction p of the output units of the previous layer
    to zero.
    
    The Dropout layer randomly sets input units to 0 with a frequency of rate 
    at each step during training time, which helps prevent overfitting. 
    Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over 
    all inputs is unchanged.
    
    Parameters
    ----------
    p : int, optional
        probability to be drop-out. The default is 0.2.
        
    """
    
    def __init__(self, p: int =0.2):

        self.dropout_rate = p
        self.input_shape = None

    def forward_pass(self, X : cp.array,
                     training : str ) -> cp.array:
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
            output array of dropout.
            
        '''
        
        if training: 
            self._mask = cp.random.uniform(0,1.0, size=X.shape) > self.dropout_rate
            return X * self._mask
        
        else:
            return X

    def backward_pass(self, accum_grad: cp.array) -> cp.array:
        '''
        
        Parameters
        ----------
        accum_grad : cp.array
            gradient with respect to weight.

        Returns
        -------
        cp.array
            gradient of dropout.
            
        '''
        
        return accum_grad * self._mask
    
    def determin_output_shape(self):
        '''
        Return input shape of this object layers
        '''
        return self.input_shape
