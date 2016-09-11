# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:25:10 2016

@author: debroissia
"""

import tensorflow as tf

def grad_inverter(grad, action, action_bounds=None):
    action_size=action.get_shape()[-1]
    if action_bounds==None:
        action_bounds = [[],[]]
        for a in range(action_size):
            action_bounds[0].append(1.0)
            action_bounds[1].append(-1.0)
    graph = action.graph
    with graph.as_default():     
        
        pmax = tf.constant(action_bounds[0], dtype = tf.float32)
        pmin = tf.constant(action_bounds[1], dtype = tf.float32)
        prange = tf.constant([x - y for x, y in zip(action_bounds[0],action_bounds[1])], dtype = tf.float32)
        pdiff_max = tf.div(-action+pmax, prange)
        pdiff_min = tf.div(action - pmin, prange)
        zeros_act_grad_filter = tf.zeros([action_size])
        grad_inverter = tf.select(tf.greater(grad, zeros_act_grad_filter), tf.mul(grad, pdiff_max), tf.mul(grad, pdiff_min))
    return grad_inverter