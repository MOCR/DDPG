# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:57:55 2016

@author: arnaud
"""

import tensorflow as tf

from DDPG.core.networks.helper.tf_session_handler import getSession 

class operate:
    def __init__(self, op, inputs=None):
        self.graph = op.graph
        self.operation = op
        self.session = getSession(self.graph)
        self.inputs = inputs
    def __call__(self, inputs_vals=None):
        return self.session.run(self.operation, feed_dict=dict(zip(self.inputs, inputs_vals)))