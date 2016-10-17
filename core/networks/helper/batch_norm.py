# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:26:51 2016

@author: debroissia
"""

import tensorflow as tf
from moments import moments


#class batch_norm:
#    
#    def __init__(self,graph, x, size, selectTrain, sess, toTarget=None, ts=0.001):
#        
#        self.sess = sess
#        self.graph=graph
#        with graph.as_default():
#            self.mean_x_train, self.variance_x_train = moments(x, [0])
#
#            #self.mean_x_ma, self.variance_x_ma = moments(self.x_splh, [0])
#
#            self.mean_x_ma = tf.Variable(tf.zeros([size]))
#            self.variance_x_ma = tf.Variable(tf.ones([size]))
#
#
 #           self.update = tf.tuple([self.variance_x_ma.assign(0.95*self.variance_x_ma+ 0.05*self.variance_x_train)] , control_inputs=[self.mean_x_ma.assign(0.95*self.mean_x_ma+ 0.05*self.mean_x_train)])[0]
 #           self.mean_x_ma_update = tf.tuple([self.mean_x_train] , control_inputs=[])[0]
 #           self.printUp = tf.Print(self.mean_x_ma_update, [selectTrain], message="selectTrain value : ")
#            self.variance_x_ma_update = tf.tuple([self.variance_x_train], control_inputs=[])[0]
#
#            def getxmau(): return self.mean_x_ma_update
#            def getxma(): return self.mean_x_ma    
#
#            def getvxmau(): return self.variance_x_ma_update
#            def getvxma(): return self.variance_x_ma
#
#            self.mean_x = tf.cond(selectTrain, getxmau, getxma)
#            self.variance_x = tf.cond(selectTrain, getvxmau, getvxma)
#
#            self.beta = tf.Variable(tf.zeros([size]))
#            self.gamma = tf.Variable(tf.ones([size]))
#
#            #tfs.tfs.session.run(tf.initialize_variables([self.beta, self.gamma]))#, self.mean_x_ma, self.variance_x_ma]))
#            self.xNorm = tf.reshape(tf.nn.batch_norm_with_global_normalization(tf.reshape(x, [-1, 1, 1, size]), self.mean_x, self.variance_x, self.beta, self.gamma, 0.01, True), [-1, size])
#
#            if toTarget!=None:
#                self.isTracking = toTarget
#                self.updateBeta = self.beta.assign(self.beta*(1-ts)+self.isTracking.beta*ts)
#                self.updateGamma = self.gamma.assign(self.gamma*(1-ts)+self.isTracking.gamma*ts)
#                self.updateTarget = tf.group(self.updateBeta, self.updateGamma)
#    def updateTarget(self):
#        self.sess.run(self.updateBeta)
#        self.sess.run(self.updateGamma)
#    def updateMeanVar(self, x):
#        pass
##        for i in range(len(x)):
##            self.x_store.append(x[i])
##        while(len(self.x_store)>=self.x_store_size):
##            self.x_store.popleft()
#        
        
def batch_norm(graph, x, trainable=True, shared_parameters=None):
    with graph.as_default():
        mean_x_train, variance_x_train = moments(x, [0], name = "moments")
        
        size = int(x.get_shape()[-1])
        parameters = []
        
        if shared_parameters != None:
            mean_x_ma = shared_parameters[0]
            variance_x_ma = shared_parameters[1]
        else:
            mean_x_ma = tf.Variable(tf.zeros([x.get_shape()[-1]]), name="mean_x_ma", trainable=False)
            variance_x_ma = tf.Variable(tf.ones([x.get_shape()[-1]]), name="var_x_ma", trainable=False)
        
        parameters.append(mean_x_ma)
        parameters.append(variance_x_ma)


        update = tf.tuple([variance_x_ma.assign(0.95*variance_x_ma+ 0.05*variance_x_train)] , control_inputs=[mean_x_ma.assign(0.95*mean_x_ma+ 0.05*mean_x_train)])[0]
        mean_x_ma_update = tf.tuple([mean_x_train] , control_inputs=[])[0]
        variance_x_ma_update = tf.tuple([variance_x_train], control_inputs=[])[0]

        def getxmau(): return mean_x_ma_update
        def getxma(): return mean_x_ma    

        def getvxmau(): return variance_x_ma_update
        def getvxma(): return variance_x_ma

        mean_x = tf.cond(tf.shape(x)[0]>1, getxmau, getxma)
        variance_x = tf.cond(tf.shape(x)[0]>1, getvxmau, getvxma)

        if shared_parameters != None:
            beta = shared_parameters[2]
            gamma = shared_parameters[3]
        else:
            beta = tf.Variable(tf.zeros([x.get_shape()[-1]]), name="beta", trainable=trainable)
            gamma = tf.Variable(tf.ones([x.get_shape()[-1]]), name="gamma", trainable=trainable)

        parameters.append(beta)
        parameters.append(gamma)

        xNorm = tf.reshape(tf.nn.batch_norm_with_global_normalization(tf.reshape(x, [-1, 1, 1, size]), mean_x, variance_x, beta, gamma, 0.01, True), [-1, size])
        
        return xNorm, parameters, update

