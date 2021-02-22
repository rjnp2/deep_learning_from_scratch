#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:31:18 2020

@author: rjn
"""

import cupy as cp
import copy
from . import Layer
from typing import Tuple
import numpy as np

class Conv2D(Layer):
    
    """A 2D Convolution Layer.

    This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. 
    
    When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g.         input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

    Parameters:
    -----------
    n_filters : int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape : Tuple[int , int]
        A tuple (filter_height, filter_width).
    input_shape : Tuple[int , int , int], optional
        The shape of the expected input of the layer. (batch_size, channels, height, width)
        Only needs to be specified for first layer in the network.
        The default is None.
    padding : str, optional
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
        The default is 'same'.
    stride : int, optional
        The stride length of the filters during the convolution over the input.
        The default is 1.
        
    """

    def __init__(self, n_filters : int,
                 filter_shape : Tuple[int , int],
                 input_shape: Tuple[int , int , int] =None, 
                 padding : str ='same', stride : int =1,
                 ):
        
        assert n_filters > 0 , "n_filters of conv2d should be greater than 0"
        self.n_filters = n_filters
        
        assert filter_shape > (0,0) , "filter_shape of conv2d should be greater than 0"
        self.filter_shape = filter_shape
        
        self.padding = padding
        self.stride = stride               
        self.input_shape = input_shape
        self.trainable = True
        self.pad = self.determine_padding()
        
    def valid_layer(self):
        
        self.output_shape = self.determin_output_shape()
        
        assert self.output_shape[1] > 0 , 'height of ouput_shape of conv2d should be greater than 0'
                            
        assert self.output_shape[2] > 0 , 'weight of ouput_shape of conv2d should be greater than 0'

    def initialize(self, optimizer ):
        # Initialize the weights
        self.W  = cp.random.uniform( -0.1 , 0.1, (self.filter_shape[0],self.filter_shape[1],
                                  self.input_shape[-1],self.n_filters))

        self.w0 = cp.random.uniform(-0.1 , 0.1, (self.n_filters))
        cp.cuda.Stream.null.synchronize()

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
            output array of convoluted.
            
        '''
         
        batch_size, _, _ ,_ = X.shape

        if training:
            self.layer_input = (X.get()).copy()
        
        X =  cp.pad( array= X, pad_width=((0, 0), 
          (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode='constant')
        cp.cuda.Stream.null.synchronize()

        h_out, w_out= self.output
        h_f, w_f = self.filter_shape 

        output = cp.zeros((batch_size, h_out, w_out, self.n_filters))
        cp.cuda.Stream.null.synchronize()

        for i in range(h_out):
            for j in range(w_out):
            
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = j * self.stride
                w_end = w_start + w_f
        
                output[:, i, j, :] = cp.sum(
                    X[:, h_start:h_end, w_start:w_end, :, cp.newaxis] *
                    self.W[cp.newaxis, :, :, :], axis=(1, 2, 3))
        
        cp.cuda.Stream.null.synchronize()

        # Calculate output
        output = output + self.w0

        cp.cuda.Stream.null.synchronize()
              
        return output

    def backward_pass(self, accum_grad : cp.array ):
        '''
        
        Parameters
        ----------
        accum_grad : cp.array
            gradient with respect to weight.

        Returns
        -------
        cp.array
            gradient of convoluted.
            
        '''

        h_out, w_out= self.output
        n, h_in, w_in, _ = self.layer_input.shape
        
        h_f, w_f = self.filter_shape 
        
        self.layer_input =  cp.pad( array=cp.asarray(self.layer_input), pad_width=((0, 0), 
          (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode='constant')

        cp.cuda.Stream.null.synchronize()

        output = cp.zeros_like(self.layer_input)
        cp.cuda.Stream.null.synchronize()

        grad_w = cp.zeros_like(self.W)
        cp.cuda.Stream.null.synchronize()

        for i in range(h_out):
            for j in range(w_out):
                
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = j * self.stride
                w_end = w_start + w_f
                
                output[:, h_start:h_end, w_start:w_end, :] += cp.sum(
                    self.W[cp.newaxis, :, :, :, :] *
                    accum_grad[:, i:i+1, j:j+1, cp.newaxis, :],
                    axis=4
                )
                cp.cuda.Stream.null.synchronize()

                grad_w+= cp.sum(
                    self.layer_input[:, h_start:h_end, w_start:w_end, :, cp.newaxis] *
                    accum_grad[:, i:i+1, j:j+1, cp.newaxis, :],
                    axis=0
                )
                cp.cuda.Stream.null.synchronize()
        
        if self.trainable:
            # n = self.layer_input.shape[0]
            # Calculate gradient w.r.t layer weights
            grad_w = grad_w / n
            grad_w0 = cp.sum( accum_grad , axis=(0, 1, 2)) / n
            
            assert grad_w.shape == self.W.shape and grad_w0.shape == self.w0.shape
            
            # Update the layer weights
            self.W , self.w0 = self.W_opt.update(w=self.W, b= self.w0 , 
                                grad_wrt_w= grad_w, grad_wrt_b =grad_w0)
            cp.cuda.Stream.null.synchronize()

        del self.layer_input , grad_w0 , grad_w
        
        return output[:, self.pad[0]:self.pad[0]+h_in, self.pad[1]:self.pad[1]+w_in, :]
    
    def determin_output_shape(self):
       
        self.output = (self.input_shape[1:3] + 2 * np.asarray(self.pad) - self.filter_shape ) // self.stride + 1
        return None ,self.output[0], self.output[1] ,self.n_filters

    def determine_padding(self) -> Tuple[int, int]:
        
        """
        Return
        ------
        output - 2 element tuple (h_pad, w_pad)
        ------------------------------------------------------------------------
        h_pad - single side padding on height of the volume
        w_pad - single side padding on width of the volume
        """
           
        # Pad so that the output shape is the same as input shape (given that stride=1)
        if self.padding == "same":
    
            filter_shape = np.asarray(self.filter_shape)        
            # Derived from:
            # output_height = (height + pad_h - filter_height) / stride + 1
            # In this case output_height = height and stride = 1. This gives the
            # expression for the padding below.
                        
            return (filter_shape - 1)//2
        
        elif self.padding == "valid":
            return (0 , 0)
        
        else:
            raise "Unsupported padding value: {self._padding}"
