# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:19:24 2016

@author: arnaud
"""

import tensorflow as tf

from DDPG.core.networks.helper.tf_session_handler import getSession 

class operation_sequence:
    
    def __init__(self, Ops, inputs=None):
        self.inputs = inputs
        previous_ops = []
        graph = None
        if not isinstance(Ops[0], list):
            graph=Ops[0].graph
        else:
            graph=Ops[0][0].graph
        self.graph = graph
        with self.graph.as_default():
            self.session = getSession(self.graph)
            for op_row in Ops:
                if not isinstance(op_row, list):
                    op_row=[op_row]
                with tf.control_dependencies(previous_ops+op_row):
                    previous_ops=[tf.no_op()]
            self.operations = previous_ops
        
    def __call__(self, inputs_vals=None):
        
        self.session.run(self.operations, feed_dict=dict(zip(self.inputs, inputs_vals)))
                    
            