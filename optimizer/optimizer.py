#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:00:11 2020

@author: rjn
"""

import cupy as cp

# Optimizers for models that use gradient based methods for finding the 
# weights that minimizes the loss.
# A great resource for understanding these methods: 

class RMSprop():
    '''
    RmsProp [tieleman2012rmsprop] is an optimizer that utilizes the magnitude 
    of recent gradients to normalize the gradients. We always keep a moving 
    average over the root mean squared (hence Rms) gradients, by which we divide
    the current gradient.
    
    
    '''
    def __init__(self, learning_rate=0.001, rho=0.9):
        
        self.learning_rate = learning_rate
        self.Eg_w = None  #Running average of the square gradients at w
        self.Eg_b = None  #Running average of the square gradients at b
        self.eps = 1e-8
        self.rho = rho

    def update(self, w,b, grad_wrt_w,grad_wrt_b):
        
        # If not initialized
        if self.Eg_w is None or self.Eg_b is None:
            self.Eg_w = cp.zeros(grad_wrt_w.shape)
            self.Eg_b = cp.zeros(grad_wrt_b.shape)

        self.Eg_w = self.rho * self.Eg_w + (1 - self.rho) * cp.square(grad_wrt_w)
        self.Eg_b = self.rho * self.Eg_b + (1 - self.rho) * cp.square(grad_wrt_b)

        # Divide the learning rate for a weight by a running average of the 
        # magnitudes of recent gradients for that weight
        
        nw =  w - self.learning_rate *  grad_wrt_w / (cp.sqrt(self.Eg_w) + self.eps)
        nb =  b - self.learning_rate *  grad_wrt_b / (cp.sqrt(self.Eg_b) + self.eps)
        
        assert nw.shape == w.shape and nb.shape == b.shape
        
        return nw,nb
