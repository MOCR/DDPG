# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:11:42 2016

@author: debroissia
"""

from network import network

class actor_network(network):
    """Base class for actor networks"""
    def action(self, state, target=False):
        """Return the actor's action for state"""
        pass
    def action_batch(self, state_batch, target=False):
        pass
    def learning_piece(self, state, act_grad):
        """add a training sample to the network's update, do not perform the update"""
        pass
    def batch(self, state, act_grad):
        pass