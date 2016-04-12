# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:44:30 2016

@author: arnaud
"""

from env import Env

class multiEnv(Env):
    
    def __init__(self, myEnv, num):
        self.protoEnv = myEnv
        self.environements = []
        self.actifs = []
        self.extern_draw = myEnv.extern_draw
        self.print_interval = myEnv.print_interval
        for i in range(num):
            self.environements.append(self.protoEnv())
            self.actifs.append(True)
    def noise_func(self):
        return 0.0
    def getActionSize(self):
        return self.environements[0].getActionSize()
    def getStateSize(self):
        return self.environements[0].getStateSize()
    def getActionBounds(self):
        return self.environements[0].getActionBounds()
    def act(self, action):
        a = 0
        ret = []
        actNoise = []
        #print action
        for i in range(len(self.actifs)):
            if self.actifs[i]:
                ac, r = self.environements[i].act([action[a]])
                ret += r
                actNoise += ac
                a += 1
        if len(ret) == 0:
            print "error ,",self.actifs
        #print ret
        return actNoise, ret
    def state(self):
        ret = []
        for i in range(len(self.actifs)):
            #print "cowboy"
            if self.actifs[i]:
                ret.append(self.environements[i].state())
        #print ret
        if len(ret) == 0:
            print "error ,",self.actifs
        return ret
    def reset(self, noise=True):
        for i in range(len(self.environements)):
            self.environements[i].reset(noise)
            self.actifs[i] = True
    def draw(self):
        self.extern_draw()
    def isFinished(self):
        ret = True
        for i in range(len(self.environements)):
            if not self.environements[i].isFinished() and self.actifs[i]:
                ret = False
            else:
                self.actifs[i] = False
        return ret
    def printEpisode(self):
        for i in range(len(self.environements)):
            self.environements[i].printEpisode()
    def performances(self):
        self.environements[0].performances()