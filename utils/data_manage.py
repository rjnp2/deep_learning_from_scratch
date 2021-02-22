#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:55:42 2020

@author: rjn
"""

# importing required library
import numpy as np

def shuffle_data(X : np.array ,
                 y: np.array,
                 seed: int = None)-> np.array:
    '''
    Random shuffle of the samples in X and 

    Parameters
    ----------
    X : np.array
         Dataset.
    y : np.array
        label data.
    seed : int, optional
        seed values. The default is None.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    np.array
        DESCRIPTION.

    '''

    if seed and type(seed) != int:
        raise TypeError('invalid data types')
    if seed:
        np.random.seed(seed)
    
    # get spaced values within a given interval of X.shape[0].
    idx = np.arange(X.shape[0])
    
    # Modify a sequence of idx by shuffling its contents.
    np.random.shuffle(idx)
    
    # return shuffle data of given X and y.
    return X[idx],y[idx]
    
def batch_iterator(X : np.array ,
                 y: np.array = None,
                 batch_size: int =64)-> np.array:
    '''
    Simple generating batch as per batch size

    Parameters
    ----------
    X : Array of int or float
        Dataset.
    y : Array of int or float
        label data. The default is None.
    batch_size : int, optional
        no of batch to be generates . The default is 64.

    Returns
    -------
    yeild of dataset of batch size.

    '''
    
    # total size of data in x 
    n_samples = X.shape[0]
    
    # checking if there is same no of data or not.
    # raise error if not.
    if y is not None and X.shape[0] != y.shape[0]:
        raise Exception(f'X and y should have same  no of data. x has {X.shape[0]} and y has {y.shape[0]}')
        
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

def train_test_split(X : np.array ,
                     y : np.array , 
                     test_size: int =0.5,
                     shuffle: str =True, 
                     seed: int =None):
    '''
    Split the data into train and test sets

    Parameters
    ----------
    X : Array of int or float
        Dataset.
    y : Array of int or float
        label data.
    test_size : float, optional
        percentage of test size. The default is 0.5.
    shuffle : Boolean, optional
        DESCRIPTION. The default is True.
    seed : int, optional
        seeding values. The default is None.

    Returns
    -------
    Spliting data into train and test sets of x and y.

    '''
    
    if shuffle:
        X, y = shuffle_data(X, y, seed)
        
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test
