# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:46:42 2016

@author: arnaud
"""

import tensorflow as tf
from DDPG.core.networks.helper.tf_session_handler import getSession 

def temporal_difference_error(net, reward_input, q_s_a, gamma = 0.99, regularization=0.0001):
    graph = reward_input.graph
    with graph.as_default():
        y_opp = reward_input + q_s_a*gamma
        temporal_diff_error = tf.pow(net.output-y_opp, 2)/tf.to_float(tf.shape(y_opp)[0]) + regularization*(tf.reduce_sum(tf.pow(net.params[-1],2)) + tf.reduce_sum(tf.pow(net.params[-2],2)))
        return temporal_diff_error
        
def target_error(net, target, regularization=0.0001):
    graph = net.graph
    with graph.as_default():
        error = tf.pow(net.output-target, 2)/tf.to_float(tf.shape(target)[0]) + regularization*(tf.reduce_sum(tf.pow(net.params[-1],2)) + tf.reduce_sum(tf.pow(net.params[-2],2)))
        return error

def minimize_error(net,error, learning_rate):
    with net.graph.as_default():
        variables = tf.all_variables()
        b_vars = []
        for v in variables:
            b_vars.append(v.name)
        adam = tf.train.AdamOptimizer(learning_rate)
        optimizer = adam.minimize(error, var_list=net.params)
        init = []
        variables = tf.all_variables()
        for v in variables:
            if not v.name in b_vars:
                init.append(v)
        getSession(net.graph).run(tf.initialize_variables(init))
        return optimizer

def gradient_output_over_tensor(net, specific_tensor):
    with net.graph.as_default():
        grad_v = tf.gradients(net.output, specific_tensor)
        grad = grad_v[0]/tf.to_float(tf.shape(grad_v[0])[0])
        return grad
    
def update_over_output_gradient(net, grad, learning_rate):
    with net.graph.as_default():
        variables = tf.all_variables()
        b_vars = []
        for v in variables:
            b_vars.append(v.name)
        params_grad = tf.gradients(net.output, net.params, -grad)          
        adam = tf.train.AdamOptimizer(learning_rate)        
        updater = adam.apply_gradients(zip(params_grad, net.params))
        init = []
        variables = tf.all_variables()
        for v in variables:
            if not v.name in b_vars:
                init.append(v)
        getSession(net.graph).run(tf.initialize_variables(init))
        return updater