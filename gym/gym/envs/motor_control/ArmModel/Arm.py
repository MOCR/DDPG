#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Corentin Arnaud

Module: Arm

Description:    
-Models an arm with two joints and six muscles
-Computes its dynamics
'''
import numpy as np
import math

class Arm(object):
    def __init__(self,armParameter, musclesparameter, dotq0):
        self.__dotq0 = dotq0
        self.armP = armParameter
        self.musclesP = musclesparameter
    
    def set_state(self, state):
        self.state = state
      
    def get_state(self):
        return self.state
      
    def setDT(self, dt):
        self.dt = dt
      
    def get_dotq_0(self):
        return np.array(self.__dotq0)

    def set_dotq_0(self, value):
        self.__dotq0 = value
        
    def setNoise(self, noise):
        self.musclesP.knoiseU=noise
        
    def jointStop(self,q):
        '''
        Articular stop for the human arm
        The stops are included in the arm parameters file
    
        Inputs:    -q: (2 or 3,1) numpy array
    
        Outputs:    -q: (2 or 3,1) numpy array
        '''
        for i in range(q.shape[0]):
            if q[i] < self.armP.lowerBounds[i]:
                q[i] = self.armP.lowerBounds[i]
            elif q[i] > self.armP.upperBounds[i]:
                q[i] = self.armP.upperBounds[i]
        return q
    
    def manipulability(self, q, target):
        J = self.jacobian(q)
        K = np.transpose(J)
        M = np.dot(J,K)
        det = np.linalg.det(M)

        return math.sqrt(det)
    
    def directionalManipulability(self, q, target):
        J = self.jacobian(q)
        #print "J", J
        K = np.transpose(J)
        #print "K", K
        M = np.dot(J,K)
        Minv= np.linalg.inv(M)

        coordHand = self.mgdEndEffector(q)

        vdir =  np.array([target[0]-coordHand[0],target[1]-coordHand[1]])
        vdir = vdir/np.linalg.norm(vdir)
        vdirt = np.transpose(vdir)

        root = np.dot(vdirt,np.dot(Minv,vdir))
       
        manip = 1/math.sqrt(root)
        return manip
    
    def getDotQAndQFromStateVector(self, state):
        '''
        Returns dotq and q from the state vector state
        
        Input:    -state: numpy array, state vector
        
        Outputs:    -dotq: numpy array
        -q: numpy array
        '''
        state=np.array(state)
        middle=state.shape[0]/2
        dotq = state[:middle]
        q = state[middle:]
        return dotq, q
