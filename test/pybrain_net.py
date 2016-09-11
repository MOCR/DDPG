#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Olivier Sigaud

Module: NeuralNet

Description: A NeuralNet in pybrain
'''

import numpy as np
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer

class NeuralNetPB():
    
    def __init__(self, inputDim, outputDim):
        '''
	Initializes class parameters
	
	Input:   

        '''
        self.inputDimension = inputDim
        self.outputDimension = outputDim
        self.net = buildNetwork(inputDim, 5, 5, outputDim)

        print "dimensions : " + str(self.inputDimension) + "x" +  str(self.outputDimension)

    def set_theta(self, theta):
        self.net.params = theta

    def set_param(self, index, value):
        self.net.params[index] = value

    def get_param(self, index):
        return self.net.params[index]

    def load_theta(self,thetaFile):
        self.net._setParameters(np.loadtxt(thetaFile))
        #print ("theta LOAD : ", self.net.params)
        return self.net.params

    def save_theta(self,fileName):
        '''
        Records theta under numpy format
        
        Input:    -fileName: name of the file where theta will be recorded
        '''
        np.savetxt(fileName, self.net.params)

    def get_action_from_state(self, state):
        '''
        Returns the action depending on the given state
        
        Input:      -state: numpy N-D array
        
        Output:     -fa_out: numpy N-D array, output approximated
        '''
        assert(len(state)==self.inputDimension), "NeuralNetPB: Bad input format"
        return self.net.activate(state)

    def get_actions_from_batch(self, states):
        '''
        Returns the actions depending on a set of states
        
        Input:      -states: array of numpy N-D arrays
        
        Output:     list of actions
        '''
        retour = []
        for i in range(len(states)):
            state = states[i]
            assert(len(state)==self.inputDimension), "NeuralNetPB: Bad input format"
            retour.append(self.net.activate(state))
        return retour

