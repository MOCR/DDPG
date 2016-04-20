# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:07:46 2016

@author: arnaud
"""

import pickle
import os.path
import matplotlib.pyplot as plt
import time

class result_log:
    def __init__(self, algo, l1, l2):
        self.algo = algo
        self.l1 = l1
        self.l2 = l2
        self.log = [[],[],[], []]
        self.firstTime = -1
    def addData(self, totStep, t, rew):
        self.log[0].append(totStep)
        self.log[1].append(t)
        self.log[2].append(rew)
        if self.firstTime == -1:
            self.log[3].append(0)
            self.firstTime = time.time()
        else:
            self.log[3].append(time.time()-self.firstTime)
    def plotTime(self):
        plt.plot(self.log[0], self.log[1])
        plt.show(block = False)
    def plotReward(self):
        plt.plot(self.log[0], self.log[2])
        plt.show(block = False)
        
    @staticmethod
    def concatLogs(logs):
        algo = logs[0].algo
        l1 = logs[0].l1
        l2 = logs[0].l2
        for l in logs:
            if l.algo != algo or l.l1 != l1 or l.l2 != l2:
                print "[WARNING] : Concatening different setups!"
        res = result_log(algo, l1, l2)
        end = float("inf")
        endi = 0
        indexs = []
        for i in range(len(logs)):
            indexs.append(0)
            if logs[i].log[0][-1]<end:
                end = logs[i].log[0][-1]
                endi = i
        i = 0
        while logs[endi].log[0][indexs[endi]]<end:
            for ii in range(len(logs)):
                if logs[ii].log[0][indexs[ii]]<logs[i].log[0][indexs[i]]:
                    i = ii
            res.addData(logs[i].log[0][indexs[i]], logs[i].log[1][indexs[i]],logs[i].log[2][indexs[i]])
            indexs[i] += 1
        return res
    @staticmethod
    def moyenLog(l, scale):
        i=0
        nxti = 0
        res = result_log(l.algo, l.l1, l.l2)
        num = 0
        sumTimes = 0
        sumRewards = 0
        numItems = 0
        setNxt = False
        while i<len(l.log[0]):
            if l.log[0][i]>(num+1)*scale:
                if numItems != 0:
                    res.addData(num*scale, sumTimes/numItems, sumRewards/numItems)
                    i = nxti
                    num+=1
                    sumTimes = 0
                    sumRewards = 0
                    numItems = 0
                    setNxt = False
            if l.log[0][i]>=(num-1)*scale:
                sumTimes += l.log[1][i]
                sumRewards += l.log[2][i]
                numItems+= 1
                if not setNxt:
                    setNxt = True
                    nxti = i
            i+=1
        return res
            
            
            
        
    def save(self, filename=None):
        if filename == None:
            i = 0
            while(os.path.exists("./results/"+self.algo+"_"+str(self.l1)+"_"+str(self.l2)+"_"+str(i)+".log")):
                i+=1
            filename = "./results/"+self.algo+"_"+str(self.l1)+"_"+str(self.l2)+"_"+str(i)+".log"
        f = open(filename, 'w')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    @staticmethod
    def load(filename):
        f = open(filename, 'r')
        ret = pickle.load(f)
        return ret
        