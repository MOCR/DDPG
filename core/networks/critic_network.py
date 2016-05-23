# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:14:03 2016

@author: debroissia
"""

from network import network

class critic_network(network):
    """Base class for a critic network"""
    def q_val(self, state, action, target=False):
        """return the estimated q value of the couple state-action"""
        pass
    def q_val_batch(self, state, action, target=False):
        pass
    def y_val_calc(self, reward, q_vals):
        pass
    def actionGradient(self, state, action):
        """return the gradient of the action"""
        pass
    def actionGradient_batch(self, state, action):
        pass
    def learning_piece(self, state, action, value):
        """add a training sample to the network's update, do not perform the update"""
        pass
    def batch(self, state, action, value):
        pass