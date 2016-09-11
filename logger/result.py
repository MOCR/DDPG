# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:07:46 2016

@author: arnaud
"""

import pickle
import os.path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import time
import numpy as np

import DDPG as ddpg

class result_log:

    def __init__(self, algo, l1, l2, idString = "", rate=0):
        self.algo = algo
        self.l1 = l1
        self.l2 = l2
        self.log = [[],[],[], []]
        self.idString = idString
        self.firstTime = -1
        self.rate=rate

    def addData(self, totStep, t, rew, realTime = None):
        self.log[0].append(totStep)
        self.log[1].append(t)
        self.log[2].append(rew)
        if realTime == None:
            if self.firstTime == -1:
                self.log[3].append(0)
                self.firstTime = time.time()
            else:
                self.log[3].append(time.time()-self.firstTime)
        else:
            self.log[3].append(realTime)

    def plotTime(self, realTime =False):
        if realTime:
            plt.plot(self.log[3], self.log[1])
        else:
            plt.plot(self.log[0], self.log[1])
        plt.show(block = False)

    def plotReward(self, realTime =False):
        if realTime:
            plt.plot(self.log[3], self.log[2])
        else:
            plt.plot(self.log[0], self.log[2])
        plt.show(block = False)

    def meanPlot(self, scale, space=None, xIndex = 0, yIndex = 1):
        i=0
        nxti = 0
        num = 0
        sumTimes = 0
        numItems = 0
        setNxt = False
        if space == None:
            space = scale
        x = []
        y = []
        error = [[], []]
        tmp = []
        while i<len(self.log[xIndex]):
            if self.log[xIndex][i]>num*space+scale:
                if numItems != 0:
                    x.append(num*space)
                    y.append(np.percentile(tmp,50))
                    error[0].append(np.percentile(tmp,25))
                    error[1].append(np.percentile(tmp,75))
                    i = nxti
                    num+=1
                    tmp = []
                    numItems = 0
                    setNxt = False
            if self.log[xIndex][i]>=num*space-scale:
                tmp.append(self.log[yIndex][i])
                numItems+= 1
                if not setNxt:
                    setNxt = True
                    nxti = i
            i+=1
        c = plt.plot(x, y, zorder = 10, label=self.algo)[0].get_color()
        plt.fill_between(x, error[0], error[1], color=c, alpha='0.25', zorder = 0)
        
        plt.show(block=False)
        
    @staticmethod
    def load_dir(path = "."):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        logs = []
        for f in files:
            if f.endswith(".log"):
                logs.append(result_log.load(path+"/"+f))
        return logs

    @staticmethod          
    def sortConcat(logs, sortIndex=0):
        sorted_logs = []
        for l in logs:
            assigned = False
            for sl in sorted_logs:
                if sl[0].algo == l.algo and sl[0].l1 == l.l1 and sl[0].l2 == l.l2 and ((not hasattr(l, "rate") and not hasattr(sl[0], "rate")) or (sl[0].rate == l.rate))  and ( (not hasattr(l, "idString") and not hasattr(sl[0], "idString")) or (sl[0].idString == l.idString)):
                    sl.append(l)
                    assigned = True
            if assigned == False:
                sorted_logs.append([l])
        concat_logs = []
        for l in sorted_logs:
            concat_logs.append(result_log.concatLogs(l, sortIndex))
        return concat_logs

    @staticmethod
    def concatLogs(logs, sortIndex = 0):
        algo = logs[0].algo
        l1 = logs[0].l1
        l2 = logs[0].l2
        if hasattr(logs[0], "rate"):
            rate = logs[0].rate
        else:
            rate = float('nan')
        if hasattr(logs[0], "idString"):
            idString = logs[0].idString
        else:
            idString = "[UNSPECIFIED]"
        for l in logs:
            if l.algo != algo or l.l1 != l1 or l.l2 != l2:
                print ("[WARNING] : Concatening different setups!")
        res = result_log(algo, l1, l2, idString, rate)
        end = float("inf")
        endi = 0
        indexs = []
        for i in range(len(logs)):
            indexs.append(0)
            if logs[i].log[sortIndex][-1]<end:
                end = logs[i].log[sortIndex][-1]
                endi = i
        i = 0
        while logs[endi].log[sortIndex][indexs[endi]]<end:
            for ii in range(len(logs)):
                if logs[ii].log[sortIndex][indexs[ii]]<logs[i].log[sortIndex][indexs[i]]:
                    i = ii
            res.addData(logs[i].log[0][indexs[i]], logs[i].log[1][indexs[i]],logs[i].log[2][indexs[i]], logs[i].log[3][indexs[i]])
            indexs[i] += 1
        return res

    @staticmethod
    def moyenLog(l, scale):
        i=0
        nxti = 0
        res = result_log(l.algo, l.l1, l.l2)
        num = 0
        sumTimes = 0
        sumRealTimes = 0
        sumRewards = 0
        numItems = 0
        setNxt = False
        while i<len(l.log[0]):
            if l.log[0][i]>(num+1)*scale:
                if numItems != 0:
                    res.addData(num*scale, sumTimes/numItems, sumRewards/numItems, sumRealTimes/numItems)
                    i = nxti
                    num+=1
                    sumTimes = 0
                    sumRewards = 0
                    sumRealTimes = 0
                    numItems = 0
                    setNxt = False
            if l.log[0][i]>=(num-1)*scale:
                sumTimes += l.log[1][i]
                sumRewards += l.log[2][i]
                sumRealTimes += l.log[3][i]
                numItems+= 1
                if not setNxt:
                    setNxt = True
                    nxti = i
            i+=1
        return res
        
    def save(self, filename=None):
        if filename == None:
            path = ddpg.__path__[0]
            if not os.path.exists(path+"/results/"+self.idString):
                try:
                    os.makedirs(path+"/results/"+self.idString)
                except OSError as exc: # Guard against race condition
                    print ("Error creating dir")
                    raise
            i = 0
            while(os.path.exists(path+"/results/"+self.idString+"/"+self.algo+"_"+str(self.l1)+"_"+str(self.l2)+"_"+str(i)+".log")):
                i+=1
            filename = path+"/results/"+self.idString+"/"+self.algo+"_"+str(self.l1)+"_"+str(self.l2)+"_"+str(i)+".log"
        print ("Log saved as : ", filename)
        f = open(filename, 'w')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def load(filename):
        f = open(filename, 'r')
        ret = pickle.load(f)
        return ret
        
