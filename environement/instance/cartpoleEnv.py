# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:11:03 2016

@author: arnaud
"""

import math
import random

from DDPG.environement.env import Env

import pybrain.rl.environments.cartpole as cp
from DDPG.logger.result_plot import result_plot

class carpoleEnv(Env):
    print_interval = 100
    def __init__(self):
        self.env = cp.JustBalanceTask()
        self.env.N = 1000
        self.env.randomInitialization = True
        self.noiseRange = 1.0
        self.om = 0
        self.alpha = 0.6
        self.beta = 0.4
        self.t = 0
        self.r = 0
        self.ep = 0
        self.perfs_mean = []
        self.perfs_max = []
        self.perfs_min = []
        self.perf_rew = []
        self.perf_noNoise = []
        self.tmp = []
        self.plot = result_plot()
    
    def state(self):
        return [self.env.getObservation()]
    def act(self, action):
        actNoise = action + self.noise_func()
        self.env.performAction(actNoise[0]*50)
        
        stt= self.env.getObservation()
        #r = 0        
        r = 1/(math.pow(stt[0],2)+math.pow(stt[1],2)+math.pow(stt[2],2)+math.pow(stt[3],2)+0.0001)
#        r = self.env.getReward()
#        if r==0:
#            r=1
#        else:
#            r=0
        #if self.isFinished() and not self.env.t>=self.env.N:
        #    r -= 100
        r -= math.pow(actNoise[0], 2.0)*0.1
        self.t += 1
        self.r += r
        return actNoise, [r]
    def reset(self, noise=True):
        self.env.reset()
        if self.ep % 10 == 0:
            self.perf_noNoise.append(self.t)
        self.ep += 1
        if self.ep % 10 == 0:
            self.noiseRange = 0
        else:
            self.noiseRange = math.pow(random.uniform(0.,1.0),2)
        self.om = 0
        self.tmp.append(self.t)
        self.perf_rew.append(self.r/self.t)
        self.t = 0
        self.r = 0
        if len(self.tmp) >=10:
            self.perfs_mean.append((sum(self.tmp)+0.0)/len(self.tmp))
            self.perfs_max.append(max(self.tmp))
            self.perfs_min.append(min(self.tmp))
            self.tmp = []
    def noise_func(self):
        self.om = self.om-self.alpha*self.om + self.beta*random.gauss(0,1)*self.noiseRange
        return self.om
    def isFinished(self):
        return self.env.isFinished()
    def getActionSize(self):
        return 1
    def getStateSize(self):
        return 4
    def getActionBounds(self):
        return [[1.2], [-1.2]]
    def printEpisode(self):
        print "Episode : " , self.ep, " steps : ", self.t, " reward : ", self.r, " noise : ", self.noiseRange
    def performances(self):
        self.plot.clear()
        #self.plot.add_row(self.perf_noNoise)        
        #self.plot.add_row(self.perf_rew)
        self.plot.add_row(self.perfs_mean)
        #self.plot.add_row(self.perfs_max)
        #self.plot.add_row(self.perfs_min)