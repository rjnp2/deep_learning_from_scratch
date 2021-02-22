#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:41:54 2020

@author: rjn
"""

from . import Layer
from typing import Tuple
import cupy as cp
import numpy as np

class MaxPooling2D(Layer):
    '''
    A parent class of MaxPooling2D
    
    Parameters
    ----------
    pool_shape : Tuple, optional
        A tuple (pool_height, pool_width).
        The default is (2, 2).
        
    padding : str, optional
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
        The default is 'valid'.
        
    stride : int, optional
        The stride length of the filters during the convolution over the input.
        The default is 2.
    

    '''

    def __init__(self, pool_shape : Tuple =(2, 2), stride : int =1, 
                 padding: str='valid'):
        
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True
        self._cache = {}      
        
    def valid_layer(self):
        
        self.output_shape = self.determin_output_shape()
        
        assert self.output_shape[1] > 0 , 'height of ouput_shape of maxpooling should be greater than 0'
                            
        assert self.output_shape[2] > 0 , 'weight of ouput_shape of maxpooling should be greater than 0'                            

    def forward_pass(self, X,training):

        if training:
            self.layer_input = X.get().copy()
        
        n, h_in, w_in, c = X.shape
        
        h_pool, w_pool = self.pool_shape
        h_out , w_out = self.output[0] , self.output[0]
        
        output = cp.zeros((n, h_out, w_out, c))
        cp.cuda.Stream.null.synchronize()

        for i in range(h_out):
            for j in range(w_out):
                
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                
                a_prev_slice = X[:, h_start:h_end, w_start:w_end, :]
                self._save_mask(x=a_prev_slice, cords=(i, j))
                cp.cuda.Stream.null.synchronize()

                output[:, i, j, :] = cp.max(a_prev_slice, axis=(1, 2))
                cp.cuda.Stream.null.synchronize()

        return output
        
    def backward_pass(self, accum_grad):
        
        output = cp.zeros_like(cp.asarray(self.layer_input))
        cp.cuda.Stream.null.synchronize()

        _, h_out, w_out, _ = accum_grad.shape
        h_pool, w_pool = self.pool_shape

        for i in range(h_out):
            for j in range(w_out):
                
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                output[:, h_start:h_end, w_start:w_end, :] += \
                    accum_grad[:, i:i + 1, j:j + 1, :] * self._cache[(i, j)]
                cp.cuda.Stream.null.synchronize()
        return output
    
    def _save_mask(self, x: cp.array, cords: Tuple[int, int]) -> None:
        
        mask = cp.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = cp.argmax(x, axis=1)

        n_idx, c_idx = cp.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self._cache[cords] = mask

    def determin_output_shape(self):

        self.output = ( self.input_shape[1:3] - np.asarray(self.pool_shape) ) // self.stride + 1
        return None, self.output[0], self.output[1],self.input_shape[-1]