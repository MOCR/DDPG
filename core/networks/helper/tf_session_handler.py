# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 11:22:58 2016

@author: arnaud
"""

import tensorflow as tf

def getSession(graph):
    if not hasattr(getSession, "lib"):
        getSession.lib = {}
    if not graph in getSession.lib:
        getSession.lib[graph] = tf.Session(graph=graph)
    return getSession.lib[graph]