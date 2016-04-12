# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:41:31 2016

@author: arnaud
"""
import math
import random

from env import Env
from mountaincar import MountainCar
#from result_plot import result_plot

class MountainCarEnv(Env):
    print_interval = 1
    def __init__(self):
        self.env = MountainCar()
        self.noiseRange = 1.0
        self.om = 0
        self.alpha = 0.6
        self.beta = 0.4
        self.t = 0
        self.r = 0
        self.ep = 0
        self.perfs = []
        self.actif = True
        #self.plot = result_plot()
    
    def state(self):
        return [self.env.getObservation()]
    def act(self, action):
        actNoise = action + self.noise_func()
        self.env.performAction(actNoise[0])
        r = self.env.getReward()
        self.t += 1
        self.r += r
        return actNoise, [r]
    def reset(self, noise=True):
        self.actif = True
        self.env.reset()
        self.om = 0
        self.perfs.append(self.t)
        self.t = 0
        self.r = 0
        self.ep += 1
        if not noise:
            self.noiseRange = 0.0
        else:
            self.noiseRange = random.uniform(0.,1.0)
    def noise_func(self):
        self.om = self.om-self.alpha*self.om + self.beta*random.gauss(0,1)*self.noiseRange
        return self.om
    def isFinished(self):
        if self.actif and not self.env.isFinished():
            return False
        else:
            self.actif = False
            return True
    def getActionSize(self):
        return 1
    def getStateSize(self):
        return 2
    def getActionBounds(self):
        return [[1.2], [-1.2]]
    def printEpisode(self):
        print "Episode : " , self.ep, " steps : ", self.t, " reward : ", self.r, " noise : ", self.noiseRange
    def performances(self):
        self.plot.add_row(self.perfs)