#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:57:07 2020

@author: rjn
"""
# importing required library

import cupy as cp

class Loss(object):
    
    def loss(self, y: cp.array , 
                   y_pred: cp.array ):
        '''
        
        Parameters
        ----------
        y : cp.array
            Ground truth value.
        y_pred : cp.array
            Predicted values.

        Returns
        -------
        NotImplementedError : TYPE
            DESCRIPTION.

        '''
        
        return NotImplementedError()

    def gradient(self, y: cp.array , 
                   y_pred: cp.array ):
        '''
        
        Parameters
        ----------
        y : cp.array
            Ground truth value.
        y_pred : cp.array
            Predicted values.

        Returns
        -------
        NotImplementedError : TYPE
            DESCRIPTION.

        '''
        raise NotImplementedError()

    def acc(self, y: cp.array , 
                   y_pred: cp.array ):
        '''
        
        Parameters
        ----------
        y : cp.array
            Ground truth value.
        y_pred : cp.array
            Predicted values.

        Returns
        -------
        NotImplementedError : TYPE
            DESCRIPTION.

        '''
        return 0
   
class mean_absolute_loss(Loss):
    '''
    Mean absolute Error (MSE) is the commonly used regression loss function. 
    MAE is the sum of absolute distances between our target variable and predicted values.
    Computes the mean absolute error between labels and predictions.
  
    The logistic  function is defined as follows:
        L(y,y_pred) =  1     
                      --- * ∑ |y - y_pred|
                       N    
                       
       `loss = mean(abs(y_true - y_pred), axis=-1)`
       
    MAE is not sensitive towards outliers and given several examples with the same 
    input feature values, and the optimal prediction will be their median target value.
    This should be compared with Mean Squared Error, where the optimal prediction is
    the mean.
    
    '''
    
    def __init__(self): 
        pass

    def loss(self, y: cp.array , 
                   y_pred: cp.array )-> cp.array:
        '''      
        Parameters
        ----------
        y : cp.array
            Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred : cp.array
            The predicted values. shape = `[batch_size, d0, .. dN]`.
    
        Returns
        -------
        cp.array
            Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.

        '''
        return cp.mean(cp.abs(y - y_pred))

    def gradient(self, y: cp.array , 
                       y_pred: cp.array ) -> cp.array:
        '''
    
        Parameters
        ----------
        y : cp.array
            Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred : cp.array
            The predicted values. shape = `[batch_size, d0, .. dN]`.
        
        Returns
        -------
        cp.array
            Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.

        '''
        return (y_pred - y)
