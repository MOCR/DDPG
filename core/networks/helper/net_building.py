# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:18:17 2016

@author: arnaud
"""

import tensorflow as tf

def create_input_layers(shapes):
    graph = tf.Graph()
    inputs = []
    with graph.as_default():
        for s in shapes:
            inputs.append(tf.check_numerics(tf.placeholder(tf.float32, s), "Input error (NaN or Inf)"))
    return inputs