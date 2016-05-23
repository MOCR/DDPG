# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:36:27 2016

@author: some dudes on github

This is a tempory solution, used to address the current issues with tf.nn.moments wich doesn't work with variable sized dimensions.

"""

import tensorflow as tf

def moments(x, axes, name=None):
  with tf.op_scope([x, axes], name, "moments"):
    x = tf.convert_to_tensor(x, name="x")
    divisor = tf.constant(1.0)
    for d in xrange(len(x.get_shape())):
      if d in axes:
        divisor *= tf.to_float(tf.shape(x)[d])
    divisor = tf.inv(divisor, name="divisor")
    axes = tf.constant(axes, name="axes")
    mean = tf.mul(tf.reduce_sum(x, axes), divisor, name="mean")
    var = tf.mul(tf.reduce_sum(tf.square(x - mean), axes),
                       divisor, name="variance")
    return mean, var