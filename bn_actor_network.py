# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:51:47 2016

@author: debroissia
"""
from actor_network import actor_network
from moments import moments
from batch_norm import batch_norm
import math

import tensorflow as tf
import tensorflow_session as tfs

class simple_actor_network(actor_network):
    l1_size = 400
    l2_size = 300
    learning_rate = 0.0001
    ts = 0.001
    """A first actor network for low-dim state"""
    def __init__(self, state_size, action_size):
        l1_size = simple_actor_network.l1_size
        l2_size = simple_actor_network.l2_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            
            self.state_input = tf.placeholder(tf.float32, [None, state_size])
    
            self.W1 = tf.Variable(tf.random_uniform([state_size, l1_size], -1/math.sqrt(state_size), 1/math.sqrt(state_size)))
            self.W2 = tf.Variable(tf.random_uniform([l1_size, l2_size], -1/math.sqrt(l1_size), 1/math.sqrt(l1_size)))
            self.W3 = tf.Variable(tf.random_uniform([l2_size, action_size], -0.0003, 0.0003))
    
            self.b1 = tf.Variable(tf.random_uniform([l1_size], -1/math.sqrt(state_size), 1/math.sqrt(state_size)))
            self.b2 = tf.Variable(tf.random_uniform([l2_size], -1/math.sqrt(l1_size), 1/math.sqrt(l1_size)))
            self.b3 = tf.Variable(tf.random_uniform([action_size], -0.0003, 0.0003))
    
            self.W1_target = tf.Variable(tf.zeros([state_size, l1_size]), trainable = False)
            self.W2_target = tf.Variable(tf.zeros([l1_size, l2_size]), trainable = False)
            self.W3_target = tf.Variable(tf.zeros([l2_size, action_size]), trainable = False)
            
            self.b1_target = tf.Variable(tf.zeros([l1_size]), trainable = False)
            self.b2_target = tf.Variable(tf.zeros([l2_size]), trainable = False)
            self.b3_target = tf.Variable(tf.zeros([action_size]), trainable = False)
            
            self.bnTrain = tf.placeholder(tf.bool, [])
            
            self.bef_x1 = tf.matmul(self.state_input,self.W1) + self.b1
            
            self.bn1 = batch_norm(self.bef_x1, l1_size, self.bnTrain,self.sess)
            
            self.x1 = tf.nn.softplus(self.bn1.xNorm)        
    
            self.bef_x2 = tf.matmul(self.x1,self.W2) + self.b2
    
            self.bn2 = batch_norm(self.bef_x2, l2_size, self.bnTrain,self.sess)
            
            self.x2 = tf.nn.tanh(self.bn2.xNorm)
            
            self.action_output = tf.matmul(self.x2,self.W3) + self.b3
            
            self.bef_x1T = tf.matmul(self.state_input,self.W1_target) + self.b1_target
            self.bn1T = batch_norm(self.bef_x1T, l1_size, self.bnTrain,self.sess, self.bn1)
            self.x1_target = tf.nn.softplus(self.bn1T.xNorm)
            self.bef_x2T = tf.matmul(self.x1_target,self.W2_target) + self.b2_target       
            
            self.bn2T = batch_norm(self.bef_x2T, l2_size, self.bnTrain,self.sess, self.bn2)
            self.x2_target = tf.nn.tanh(self.bn2T.xNorm)
            self.action_output_target = tf.matmul(self.x2_target,self.W3_target) + self.b3_target
            
            self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
            self.params = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3, self.bn1.gamma, self.bn1.beta, self.bn2.beta, self.bn2.gamma]
            self.params_grad = tf.gradients(self.action_output, self.params, -self.action_gradient)      
            
            self.adam = tf.train.AdamOptimizer(simple_actor_network.learning_rate)        
            self.optimizer = tf.train.GradientDescentOptimizer(simple_actor_network.learning_rate)
            self.updater = self.adam.apply_gradients(zip(self.params_grad, self.params))        
            
            init = tf.initialize_all_variables()                
            self.sess.run(init)
            
            self.sess.run([self.W1_target.assign(self.W1),
                           self.W2_target.assign(self.W2),
                           self.W3_target.assign(self.W3),
                           self.b1_target.assign(self.b1),
                           self.b2_target.assign(self.b2),
                           self.b3_target.assign(self.b3) ])      
            
            self.upTargW1 = self.W1_target.assign(self.W1_target*(1-simple_actor_network.ts)+ self.W1*(simple_actor_network.ts))
            self.upTargW2 = self.W2_target.assign(self.W2_target*(1-simple_actor_network.ts)+ self.W2*(simple_actor_network.ts))        
            self.upTargW3 = self.W3_target.assign(self.W3_target*(1-simple_actor_network.ts)+ self.W3*(simple_actor_network.ts))
            
            self.upTargb1 = self.b1_target.assign(self.b1_target*(1-simple_actor_network.ts)+ self.b1*(simple_actor_network.ts))
            self.upTargb2 = self.b2_target.assign(self.b2_target*(1-simple_actor_network.ts)+ self.b2*(simple_actor_network.ts))
            self.upTargb3 = self.b3_target.assign(self.b3_target*(1-simple_actor_network.ts)+ self.b3*(simple_actor_network.ts))
            
            self.batch_state = []
            self.batch_actgrad = []

    def action(self, target=False):
        """Return the actor's action for state"""
        ret = 0
        if(target):
            ret = self.sess.run(self.action_output_target, feed_dict={self.state_input: [state], self.bnTrain : False})[0]
            #self.bn1T.updateMeanVar()
            #self.bn2T.updateMeanVar(self.sess.run(self.bef_x2T, feed_dict={self.state_input: [state], self.bn1T.selectTrain: self.bn1T.infer, self.bn1T.selectUpdate: self.bn1T.update, self.bn2T.selectTrain: self.bn2T.infer, self.bn2T.selectUpdate: self.bn2T.noUpdate, self.bn2T.x_splh: self.bn2T.x_store}))
        else:
            ret = self.sess.run(self.action_output, feed_dict={self.state_input: [state], self.bnTrain : False})[0]
            #self.bn1.updateMeanVar()
            #self.bn2.updateMeanVar(self.sess.run(self.bef_x2, feed_dict={self.state_input: [state], self.bn1T.selectTrain: self.bn1T.infer, self.bn1T.selectUpdate: self.bn1T.update, self.bn2T.selectTrain: self.bn2T.infer, self.bn2T.selectUpdate: self.bn2T.noUpdate, self.bn2T.x_splh: self.bn2T.x_store}))
        return ret
        
    def action_batch(self, state_batch, target=False):
        if(target):
            return self.sess.run(self.action_output_target, feed_dict={self.state_input: state_batch,  self.bnTrain : False})
        return self.sess.run(self.action_output, feed_dict={self.state_input: state_batch, self.bnTrain : False})
        
    def learning_piece(self, state, act_grad):
        """add a training sample to the network's update, do not perform the update"""
        self.batch_state.append(state)
        self.batch_actgrad.append(act_grad)
    def batch(self, state, act_grad):
        self.batch_state = state
        self.batch_actgrad = act_grad
    def update(self):
        self.sess.run([self.updater, self.bn1.update, self.bn2.update, self.bn1T.update, self.bn2T.update], feed_dict={self.state_input: self.batch_state, self.action_gradient: self.batch_actgrad, self.bnTrain : True})
        #del self.batch_state[:]
        #del self.batch_actgrad[:]
    def updateTarget(self):
        self.sess.run([self.upTargW1,
                       self.upTargW2,
                       self.upTargW3,
                       self.upTargb1,
                       self.upTargb2,      
                       self.upTargb3,
                       self.bn1T.updateTarget,
                       self.bn2T.updateTarget])
        
#        self.bn1T.updateTarget()
#        self.bn2T.updateTarget()
        