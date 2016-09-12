#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Thomas Beucher

Module: createVectorUtil

Description: creates a vector from data
'''

import numpy as np

def createVector(*data):
    '''
    creates vector from data given
    
    Input:    -tuple of n data
    
    Output:    -vector, numpy array of dimension (n,1)
    '''
    dimVect = len(data)
    Vector = [x for x in data]
    return np.asarray(Vector).reshape((dimVect, 1))


