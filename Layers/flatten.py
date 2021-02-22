#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:46:02 2020

@author: rjn
"""

from . import Layer
import numpy as np

class Flatten(Layer):
    
    """ 
    Turns a multidimensional matrix into two-dimensional
    
    Parameters
    ----------
    input_shape : TYPE, optional
        shape of data . The default is None.
    
    """
    
    def __init__(self, input_shape : tuple =None):
        
        self.input_shape = input_shape

    def forward_pass(self, X,training):
        '''
        
        Parameters
        ----------
        X : cp.array
            array of prevoius data.

        Returns
        -------
        cp.array
            flatten output array.
            
        '''
        
        self.prev_shape = X.shape
        return (X.reshape((X.shape[0], -1))).astype('float32')

    def backward_pass(self, accum_grad):
        '''
        
        Parameters
        ----------
        accum_grad : cp.array
            gradient with respect to weight.

        Returns
        -------
        cp.array
            unflatten array.
            
        '''
        return accum_grad.reshape(self.prev_shape)

    def determin_output_shape(self):
        '''
        Return input shape of this object layers
        '''
        return (None , np.prod(self.input_shape[1:]))
