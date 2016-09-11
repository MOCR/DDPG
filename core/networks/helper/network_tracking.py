# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:41:03 2016

@author: arnaud
"""
import tensorflow as tf
from DDPG.core.networks.helper.tf_session_handler import getSession 

def track_network(net, copy, tracking_speed):
    operations = []
    for i in range(len(net.params)):
        op = copy.params[i].assign(copy.params[i]*(1-tracking_speed)+ net.params[i]*(tracking_speed))
        operations.append(op)
    return operations
    
def copy_network(net,copy):
    graph = net.params[0].graph
    copy_op = track_network(net,copy, 1)
    getSession(graph).run(copy_op)
    