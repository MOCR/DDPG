# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:26:51 2016

@author: debroissia
"""

import tensorflow as tf
from moments import moments
import tensorflow_session as tfs

from collections import deque

class batch_norm:
    
    def __init__(self, x, size, selectTrain, sess, toTarget=None, ts=0.001):
        
        self.sess = sess
        self.mean_x_train, self.variance_x_train = moments(x, [0])
        
        #self.mean_x_ma, self.variance_x_ma = moments(self.x_splh, [0])
        
        self.mean_x_ma = tf.Variable(tf.zeros([size]))
        self.variance_x_ma = tf.Variable(tf.ones([size]))

        
        self.update = tf.tuple([self.variance_x_ma.assign(0.95*self.variance_x_ma+ 0.05*self.variance_x_train)] , control_inputs=[self.mean_x_ma.assign(0.95*self.mean_x_ma+ 0.05*self.mean_x_train)])[0]
        self.mean_x_ma_update = tf.tuple([self.mean_x_train] , control_inputs=[])[0]
        self.printUp = tf.Print(self.mean_x_ma_update, [selectTrain], message="selectTrain value : ")
        self.variance_x_ma_update = tf.tuple([self.variance_x_train], control_inputs=[])[0]

        def getxmau(): return self.mean_x_ma_update
        def getxma(): return self.mean_x_ma    
        
        def getvxmau(): return self.variance_x_ma_update
        def getvxma(): return self.variance_x_ma
        
        self.mean_x = tf.cond(selectTrain, getxmau, getxma)
        self.variance_x = tf.cond(selectTrain, getvxmau, getvxma)
        
        self.beta = tf.Variable(tf.zeros([size]))
        self.gamma = tf.Variable(tf.ones([size]))
        
        #tfs.tfs.session.run(tf.initialize_variables([self.beta, self.gamma]))#, self.mean_x_ma, self.variance_x_ma]))
        self.xNorm = tf.reshape(tf.nn.batch_norm_with_global_normalization(tf.reshape(x, [-1, 1, 1, size]), self.mean_x, self.variance_x, self.beta, self.gamma, 0.01, True), [-1, size])
            
        if toTarget!=None:
            self.isTracking = toTarget
            self.updateBeta = self.beta.assign(self.beta*(1-ts)+self.isTracking.beta*ts)
            self.updateGamma = self.gamma.assign(self.gamma*(1-ts)+self.isTracking.gamma*ts)
            self.updateTarget = tf.group(self.updateBeta, self.updateGamma)
    def updateTarget(self):
        self.sess.run(self.updateBeta)
        self.sess.run(self.updateGamma)
    def updateMeanVar(self, x):
        pass
#        for i in range(len(x)):
#            self.x_store.append(x[i])
#        while(len(self.x_store)>=self.x_store_size):
#            self.x_store.popleft()
        
        
        
        