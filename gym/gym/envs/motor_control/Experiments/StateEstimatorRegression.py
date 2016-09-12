#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Corentin Arnaud

Module: StateEstimator

Description: Used to estimate the current state, reproducing the human control motor delay.
'''
import numpy as np
import random as rd
import os
from Regression.NeuralNet import NeuralNet
from Regression.NeuralNetTF import NeuralNetTF
from TrainStateEstimator import NeuraNetParameter



#from ArmModel.MuscularActivation import getNoisyCommand

def isNull(vec):
    for el in vec:
        if el!=0:
            return False
    return True

class StateEstimatorRegression:
    
    def __init__(self, dimState, dimCommand, delay, arm):
        '''
    	Initializes parameters to uses the function implemented below
    	
    	inputs:		-dimCommand: dimension of the muscular activation vector U, int (here, 6)
    			-delay: delay in time steps with which we give the observation to the filter, int
    			-arm, armModel, class object
    	'''
        self.name = "StateEstimator"
        self.dimState = dimState
        self.dimCommand = dimCommand
        self.delay = delay
        self.arm = arm
        para= NeuraNetParameter(1,"")
        self.regression = NeuralNetTF(para)
        self.regression.setTheta(np.loadtxt(para.path+para.thetaFile+".theta"))
        self.regressionInput=np.empty((para.inputDim))

    def initStore(self, state):
        '''
    	Initialization of the observation storage
    
    	Input:		-state: the state stored
    	'''
        self.stateStore = np.zeros((self.delay,self.dimState))
        self.commandStore = np.zeros((self.delay,self.dimCommand))
        #print ("InitStore:", self.stateStore)
        self.currentEstimState = state
    
    def storeInfo(self, state, command):
        '''
    	Stores the current state and returns the delayed state
    
    	Input:		-state: the state to store
    	'''
        self.stateStore[1:]=self.stateStore[:-1]
        self.commandStore[1:]=self.commandStore[:-1]
        self.stateStore[0]=state
        self.commandStore[0]=command
        #print ("After store:", self.stateStore)
        #print ("retour:", self.stateStore[self.delay-1])
        return self.stateStore[self.delay-1]
    
    def getEstimState(self, state, command):
        '''
    	Function used to compute the next state approximation with the filter
    
    	Inputs:		-state: the state to feed the filter, numpy array of dimension (x, 1), here x = 4
                    -command: the noiseless muscular activation vector U, numpy array of dimension (x, 1), here x = 6
    
    	Output:		-stateApprox: the next state approximation, numpy array of dimension (x, 1), here x = 4
    	'''
        #store the state of the arm to feed the filter with a delay on the observation
        inferredState = self.storeInfo(state, command)
        if isNull(inferredState):
            self.regressionInput[:self.dimState]=self.currentEstimState
            self.regressionInput[self.dimState:]=command

            self.currentEstimState = self.regression.computeOutput(self.regressionInput)
            return self.currentEstimState
        for i in range (self.delay):
            U = self.commandStore[self.delay-i-1]
            self.regressionInput[:self.dimState]=inferredState
            self.regressionInput[self.dimState:]=U
            inferredState = self.regression.computeOutput(self.regressionInput)
        '''
        qdot,q = getDotQAndQFromStateVector(state)
        speed = self.arm.cartesianspeed(state)
        for i in range(2,4):
            inferredState[i] = inferredState[i]*(1+ np.random.normal(0,0.01*speed))
        '''

        self.currentEstimState = inferredState
        return self.currentEstimState

    
    def debugStore(self):
        state = np.array([1,2,3,4])
        self.initObsStore(state)
        for _ in range(5):
            tmpS = [rd.random() for _ in range(4)]
            tmpU = [rd.random() for _ in range(6)]
            self.storeInfo(tmpS,tmpU)
            
    
    
