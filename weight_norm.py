# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:27:11 2016

@author: debroissia
"""

import tensorflow as tf
import math

class weight_norm:
    
    def __init__(self, lin, lout, iniRange, graph= None):
        
        if graph!=None:
            with graph.as_default():
                
                self.v = tf.Variable(tf.random_uniform([lin, lout], iniRange[0], iniRange[1]))
                self.g = tf.Variable(tf.random_uniform([lout], -1.0,1.0))
                self.pow2 = tf.fill([lin, lout],2.0)
                self.v_norm = tf.sqrt(tf.reduce_sum(tf.pow(self.v, self.pow2),0))
                self.tile_div = tf.tile(tf.expand_dims(tf.div(self.g, self.v_norm),0),[lin, 1])
                self.w = tf.mul(self.tile_div, self.v)
        else:
            self.v = tf.Variable(tf.random_uniform([lin, lout], -1/math.sqrt(lin), 1/math.sqrt(lin)))
            self.g = tf.Variable(tf.random_uniform([lout], -1.0,1.0))
            self.pow2 = tf.fill([lin, lout],2.0)
            self.v_norm = tf.sqrt(tf.reduce_sum(tf.pow(self.v, self.pow2),0))
            self.tile_div = tf.tile(tf.expand_dims(tf.div(self.g, self.v_norm),0),[lin, 1])
            self.w = tf.mul(self.tile_div, self.v)