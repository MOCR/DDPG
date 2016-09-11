# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:40:58 2016

@author: sigaud
"""
import numpy as np

import cma
import sys
import math

from DDPG.test.helpers.draw import draw_policy, run
from DDPG.core.helpers.Chrono import Chrono

import matplotlib.pyplot as plt

from simple_keras_net import Keras_NN

import gym

#env = gym.make('Pendulum-v0')
layer_nb = 2
sigma = 0.5

class CMA():

    def __init__(self):
        '''
	Input:
        '''
        self.env = gym.make('MountainCarContinuous-v0')
        self.env.reset(False)
        self.controller = Keras_NN(2,1)
        self.nb_episodes = 0
        self.options = cma.CMAOptions()
        self.options['maxiter']=200
        self.options['popsize']=50
        self.options['CMA_diagonal']=True
        self.options['verb_log']=50
        self.options['verb_disp']=1
        self.best_perf = -10000.0

    def run_episode(self,x):
        self.controller.set_all_weights_from_vector(layer_nb,x)
        self.nb_episodes += 1
        perf = run(self.controller, self.env, False, False)
        if (perf>self.best_perf):
            self.best_perf=perf
            self.controller.save_theta('cma_agents/best_agent.theta'+str(perf))
            print('episode',self.nb_episodes, 'perf', perf)
        return -perf

    def call_cma(self):
        x=  self.controller.get_weights_as_vector(layer_nb)
        print ('x',x)
        fx = cma.fmin(self.run_episode, x, sigma, options = self.options)
        
