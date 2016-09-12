#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Olivier Sigaud

Module: NeuralNet

Description: A NeuralNet in keras
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.constraints import maxnorm

class Keras_NN():
    
    def __init__(self, inputDim, outputDim):
        '''
	Input:

        '''
        self.inputDim=inputDim 
        self.model = Sequential()
        self.model.add(Dense(5, input_dim=inputDim))
        self.model.add(Activation("tanh"))
        self.model.add(Dense(5))
        self.model.add(Activation("tanh"))
#       self.model.add(Dense(outputDim,W_constraint=maxnorm(m=0.3)))
        self.model.add(Dense(outputDim))
        self.model.add(Activation("tanh"))
        self.model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
        print self.model.summary()

    def set_theta(self, theta):
        self.model.set_weights(theta)

    def get_nb_weights(self,layer_nb):
        layer = self.model.layers[layer_nb]
        all_params = layer.get_weights()
        weights = all_params[0]
        return len(weights)*len(weights[0])

    def get_weight(self,layer_nb,index):
        return self.model.layers[layer_nb].get_weights()[0][index]

    def get_weights(self,layer_nb):
        return self.model.layers[layer_nb].get_weights()[0]

    def get_biases(self,layer_nb):
        return self.model.layers[layer_nb].get_weights()[1]

    def get_weights_as_vector(self,layer_nb):
        return self.model.layers[layer_nb].get_weights()[0].reshape(1, self.get_nb_weights(layer_nb))[0]

    def get_all_params_as_vector(self,layer_nb):
        return self.model.layers[layer_nb].get_weights()[0].reshape(1, self.get_nb_weights(layer_nb))[0]+self.model.layers[layer_nb].get_weights()[1]

    def get_all_params_as_vector(self):
        vec = [] 
        for layer_nb in (range(len(self.model.layers))):
            if (len(get_weights(self,layer_nb))>0):
                vec = vec + self.model.layers[layer_nb].get_weights()[0].reshape(1, self.get_nb_weights(layer_nb))[0]+self.model.layers[layer_nb].get_weights()[1]
        return vec

    def set_param(self, layer_nb, weight_index, value):
        layer = self.model.layers[layer_nb]
        all_params = layer.get_weights()
        weights = all_params[0]
        biases = all_params[1]
        size = len(weights[0])
        num_tab = int(weight_index/size)
        rest=weight_index-num_tab*size
        '''
        print ('weight_index',weight_index,'value',value)
        print ('num_tab',num_tab,'rest',rest)
        print ('old tab',weights[num_tab])
        print ('old weight',weights[num_tab][rest])
        print ('weights before',weights)
        '''
        weights[num_tab][rest]=value
#        print ('weights after',weights)
        all_params[0]=weights
        self.model.layers[layer_nb].set_weights(all_params)

    def get_param(self, layer_nb, weight_index):
        size = len(self.model.layers[layer_nb].get_weights()[0][0])
        num_tab = int(weight_index/size)
        rest=weight_index-num_tab*size
        param = self.model.layers[layer_nb].get_weights()[0][num_tab][rest]
        print('param',param)
        return param

    def set_all_weights_from_vector(self, layer_nb, vec):
        layer = self.model.layers[layer_nb]
        all_params = layer.get_weights()
        weights = all_params[0]
        new_weights = vec.reshape(len(weights), len(weights[0]))
        all_params[0]=new_weights
        self.model.layers[layer_nb].set_weights(all_params)

    def load_theta(self,thetaFile):
        self.model.load_weights(thetaFile)
        return self.model.get_weights()

    def save_theta(self,fileName):
        '''
        Records theta under numpy format
        
        Input:    -fileName: name of the file where theta will be recorded
        '''
        self.model.save_weights(fileName)

    def get_action_from_state(self, state):
        '''
        Returns the action depending on the given state
        
        Input:      -state: numpy N-D array
        
        Output:     -fa_out: numpy N-D array, output approximated
        '''
        assert(len(state)==self.inputDim), "Keras_NN: Bad input format"
        state = state.reshape((-1,self.inputDim))
        prediction = self.model.predict(state,batch_size=1)[0]
#        print('prediction',prediction)
        return prediction

    def get_actions_from_batch(self, states):
        '''
        Returns the actions depending on a set of states
        
        Input:      -states: array of numpy N-D arrays
        
        Output:     list of actions
        '''
        retour = []
        for i in range(len(states)):
            state = np.array(states[i])
            assert(len(state)==self.inputDim), "Keras_NN: Bad input format"
            state = state.reshape((-1,self.inputDim))
            prediction = self.model.predict(state,batch_size=1)[0]
            retour.append(prediction)
        return retour

