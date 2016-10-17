# -*- coding: utf-8 -*-
"""

@author: sigaud
"""
import random

class noise_generator(object):
    """
    A noise generator from adding noise to actions
    """
    def __init__(self, size, logger = None):
        self.noise = 0
        self.alpha = 0.6
        self.beta = 0.1
        self.om = []
        for i in range(size):
            self.om.append(0.0)

#    def sample(self):
#        '''
#        Ornstein-Uhlenbeck random Process
#        '''
#        #x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
#        self.x_prev = x
#        self.n_steps += 1
#        return x
    def increase_noise(self):
        self.beta = self.beta*1.02

    def decrease_noise(self):
        self.beta = self.beta*0.7
    def randomRange(self):
        self.beta = random.uniform(0.0,1.0)

    def get_sample(self, i):
        self.om[i] = self.om[i]-self.alpha*self.om[i] + self.beta*random.gauss(0,1)
        return self.om[i]

    def add_noise(self,action_vector):
        noisy_action = []
        for i in range(len(action_vector)):
            noisy_action.append(action_vector[i]+self.get_sample(i))
        return noisy_action
