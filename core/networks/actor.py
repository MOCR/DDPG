# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:51:47 2016

@author: debroissia
"""

from DDPG.core.networks.helper.fully_connected_layer import fully_connected_layer

import tensorflow as tf

class actor:
    """The actor part of ddpg
        parameters for initialization :
        state_size : the size of the state vectors
        action_size : the size of the expected output vector
        l1_size : the size of the first hidden layer
        l2_size : the size of the second hidden layer
        function : the configuration of the networks functions (see fully_connected_layer)
        weight_init_range : the ranges for random initialization of each layer's weights
    """
    def __init__(self, 
                 state_size, 
                 action_size,
                 l1_size,
                 l2_size,
                 f=[tf.nn.softplus, tf.nn.softplus, tf.nn.tanh],
                 weight_init_range=[None,None,[-0.0003,0.0003]],
                 trainable=True):
        
        with tf.name_scope('actor_init'):
        
            self.state_input = tf.placeholder(tf.float32, [None, state_size])
            self.net = fully_connected_layer(self.state_input, 
                                             [l1_size, l2_size, action_size], 
                                             function=f,
                                             weight_init_range=weight_init_range)
            self.action_output = self.net.output
            
    def init_training(self, action_size, learning_rate):
        """ 
        init all gradient computations
        only used for the actor, not for the target actor
        parameters for initialization :
        learning_rate : the learning rate applied when updating the network
        """
        with tf.name_scope('actor_training'):
            self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
            self.params_grad = tf.gradients(self.action_output, self.net.params, -self.action_gradient) 
            
            self.adam = tf.train.AdamOptimizer(learning_rate)        
            self.updater = self.adam.apply_gradients(zip(self.params_grad, self.net.params))        
                            
    def get_action_from_state(self, sess, state):
        """
        Return the actor's action for a state
        """
        action = sess.run(self.action_output, feed_dict={self.state_input: [state]})
        return action[0]
        
    def get_actions_from_batch(self, sess, state_batch):
        """
        Return the actor's action for a batch of states
        """
        return sess.run(self.action_output, feed_dict={self.state_input: state_batch})

    def update(self, sess, state, act_grad):
        """ 
        update the network
        only used for the actor, not for the target actor
        """
        sess.run(self.updater, feed_dict={self.state_input: state, self.action_gradient: act_grad})
