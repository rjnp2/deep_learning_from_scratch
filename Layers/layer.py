#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:13:41 2020

@author: rjn
"""

class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.__class__.__name__
    
    def valid_layer(self):
        
        return None 

    def parameters(self):
        """ The number of trainable parameters used by the layer """
        return 0

    def load_parameters(self):
        return None

    def set_weights(self,weights):
        raise NotImplementedError()

    def forward_pass(self, X,training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        """ Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer. """
        raise NotImplementedError()

    def determin_output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()
