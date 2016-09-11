# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:44:14 2016

@author: debroissia
"""

from DDPG.core.networks.helper.fully_connected_layer import fully_connected_layer

import tensorflow as tf
from DDPG.core.minibatch import minibatch

class critic:
    """
    The critic part of ddpg
        parameters for initialization:
        state_size: the size of the state vectors
        action_size: the size of the expected output vector
        l1_size: the size of the first hidden layer
        l2_size: the size of the second hidden layer
        function: the configuration of the networks functions (see fully_connected_layer)
        input_layers_connections: where each input vector schould be connected (the first one is states and second one is actions)
        weight_init_range: the ranges for random initialisation of each layer's weights
    """

    def __init__(self, 
                 state_size, 
                 action_size, 
                 l1_size,
                 l2_size,
                 function = [tf.nn.softplus, tf.nn.tanh, None], 
#question: the importance of the trick below should be explained in a paper
                 input_layers_connections=[0,1],
#question: why set these ranges? Was it tuned?
                 weight_init_range=[None, None,[-0.0003,0.0003]],
                 trainable=True):
        
        with tf.name_scope('critic_init'):
            
            self.state_input = tf.placeholder(tf.float32, [None, state_size])
            self.action_input = tf.placeholder(tf.float32, [None, action_size])
            
            self.net= fully_connected_layer([self.state_input, self.action_input],
                                            [l1_size, l2_size, 1], 
                                            function = function, 
                                            input_layers_connections=input_layers_connections,
                                            weight_init_range=weight_init_range)
            
            self.qval_output = self.net.output
    
    def init_training(self, target_output, learning_rate, gamma, regularization):
        """ 
        init all gradient computations
        only used for the actor, not for the target actor
        parameters for initialization:
        learning_rate: the learning rate applied when updating the network
        gamma: the discount factor for the discounted reward calculation
        """
        with tf.name_scope('critic_training'):
            self.act_grad_v = tf.gradients(self.qval_output, self.action_input)
            self.act_grad = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])]
            
            self.rewards = tf.placeholder(tf.float32, [None, 1])
            self.y_opp = self.rewards + target_output*gamma
#question: why this 0.0001 regularization factor? Was it tuned?
            self.temporal_diff_error = tf.pow(self.qval_output-self.y_opp, 2)/tf.to_float(tf.shape(self.y_opp)[0]) + regularization*(tf.reduce_sum(tf.pow(self.net.params[-1],2)) + tf.reduce_sum(tf.pow(self.net.params[-2],2)))
            
            self.adam = tf.train.AdamOptimizer(learning_rate)       
            self.optimizer = self.adam.minimize(self.temporal_diff_error)
            
    def get_q_val(self, sess, state, action):
        """
        return the estimated q value of one state-action pair
        never used
        """
        return sess.run(self.qval_output, feed_dict={self.state_input: [state], self.action_input: [action]})[0][0]

    def get_q_val_batch(self, sess, state, action):
        """
        return the estimated q value of a batch of state-action pairs
        never used
        """
        return sess.run(self.qval_output, feed_dict={self.state_input: state, self.action_input: action})

    def get_actionGradient(self, state, action):
        """
        return the gradient with respect to the action for a state
        never used
        """
        return self.sess.run(self.act_grad, feed_dict={self.state_input: [state], self.action_input: [action]})[0][0]

    def get_actionGradient_batch(self, sess, state, action):
        """
        return the gradient with respect to the action over a batch of states
        """
        return sess.run(self.act_grad, feed_dict={self.state_input: state, self.action_input: action})[0]

    def update(self, sess, minibatch, target_state_input, target_action_input, action_target):
        """
        update the weights of the critic
        """
        sess.run(self.optimizer, feed_dict={self.state_input: minibatch.states,
                                            self.action_input: minibatch.actions, 
                                            self.rewards: minibatch.rewards, 
                                            target_state_input: minibatch.next_states,
                                            target_action_input: action_target})
            
    def get_output(self):
        """
        return the output layer of the network
        used to get target critic in gradient calculation
        """
        return self.net.output
            
    def get_action_input(self):
        """
        return the action input layer of the network
        used to get target critic in gradient calculation
        """
        return self.action_input
           
    def get_state_input(self):
        """
        return the state input layer of the network
        used to get target critic in gradient calculation
        """
        return self.state_input
