# -*- coding: utf-8 -*-
"""

@author: sigaud
"""
import random

class noise_generator(object):
    """
    A noise generator from adding noise to actions
    """
    def __init__(self, logger = None):
        self.noiseRange = 1.0
        self.noise = 0
        self.alpha = 0.6
        self.beta = 0.4

    def sample(self):
        '''
        Ornstein-Uhlenbeck random Process
        '''
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def update_noise(self):
        self.noise = self.get_sample()

    def get_sample(self):
        return self.noise-self.alpha*self.noise + self.beta*random.gauss(0,1)*self.noiseRange

    def add_noise(self,action_vector):
        noisy_action = []
        self.update_noise()
        for i in range(len(action_vector)):
            noisy_action.append(action_vector[i]+self.get_sample())
        return noisy_action
