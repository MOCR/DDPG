#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Thomas Beucher

Module: Experiments

Description: Class used to generate all the trajectories of the experimental setup and also used for CMAES optimization
'''


import numpy as np
import time
#from Utils.ThetaNormalization import normalization, unNormalization


from GlobalVariables import pathDataFolder

from TrajMaker import TrajMaker
from Utils.FileWriting import checkIfFolderExists, findDataFilename, writeArray, find_best_theta_file
from multiprocess.pool import Pool
from functools import partial
#------------------------------------------------------------------------------

class Experiments:
    def __init__(self, rs, sizeOfTarget, saveTraj, foldername, thetafile, popSize, period, estim="Inv"):
        '''
    	Initializes parameters used to run functions below
    
    	Inputs:
     	'''
        self.rs = rs
        self.name = "Experiments"
        self.call = 0
        self.dimState = rs.inputDim
        self.dimOutput = rs.outputDim
        self.numberOfRepeat = rs.numberOfRepeatEachTraj
        self.foldername = foldername
        self.tm = TrajMaker(rs, sizeOfTarget, saveTraj, thetafile, estim)
        self.posIni = np.loadtxt(pathDataFolder + rs.experimentFilePosIni)
        if(len(self.posIni.shape)==1):
            self.posIni=self.posIni.reshape((1,self.posIni.shape[0]))
        self.costStore = []
        self.cost12Store=[]
        self.CMAESCostStore = []
        self.CMAESTimeStore = []
        self.trajTimeStore = []
        self.bestCost = -10000.0
        self.lastCoord = []
        self.popSize = popSize
        self.period = period
        
    def printLastCoordInfo(self):
        vec = np.array(self.lastCoord)
        print ("moyenne : "+ str(np.mean(vec)))
        print ("min : " + str(np.min(vec)))
        print ("max :" + str(np.max(vec)))
    

    def initTheta(self, theta):
        '''
     	Input:		-theta: controller ie vector of parameters, numpy array
    	'''
        self.theta=theta
        self.tm.setTheta(self.theta)

    def saveCost(self):
        ''' 
        filename = findDataFilename(self.foldername+"Cost/","traj",".cost")
        filenameTime = findDataFilename(self.foldername+"TrajTime/","traj",".time")
        filenameX = findDataFilename(self.foldername+"finalX/","x",".last")
        np.savetxt(filename, self.costStore)
        np.savetxt(filenameTime, self.trajTimeStore)
        np.savetxt(filenameX, self.lastCoord)
        '''
        writeArray(self.costStore,self.foldername+"Cost/","traj",".cost")
        writeArray(self.cost12Store,self.foldername+"CostU12/","traj",".cost")
        writeArray(self.trajTimeStore, self.foldername+"TrajTime/","traj",".time")
        writeArray(self.lastCoord, self.foldername+"finalX/","x",".last")
        
    def setNoise(self, noise):
        self.tm.setnoise(noise)
         
    def runOneTrajectory(self, x, y):
        #self.tm.saveTraj = True
        cost, trajTime, lastX = self.tm.runTrajectory(x, y, self.foldername)
        #cost, trajTime, lastX = self.tm.runTrajectoryOpti(x, y)
        #print "Exp local x y cost : ", x, y, cost
        if lastX != -1000:
            self.lastCoord.append(lastX)
        return cost, trajTime
            
    def runRichTrajectories(self, repeat):
        globCost = []
        xy = np.loadtxt(pathDataFolder + "PosCircu540")
        #xy = np.loadtxt(pathDataFolder + "PosSquare")
        for el in xy:
            costAll, trajTimeAll = np.zeros(repeat), np.zeros(repeat)
            for i in range(repeat):
                costAll[i], trajTimeAll[i]  = self.runOneTrajectory(el[0], el[1]) 
            meanCost = np.mean(costAll)
            meanTrajTime = np.mean(trajTimeAll)
            self.costStore.append([el[0], el[1], meanCost])
            self.trajTimeStore.append([el[0], el[1], meanTrajTime])
            globCost.append(meanCost)
        return np.mean(globCost)
            
    def runTrajectoriesForResultsGeneration(self, repeat):
        globMeanCost=0.
        globTimeCost=0.
        for xy in self.posIni:
            costAll, trajTimeAll, costU12 = np.zeros(repeat), np.zeros(repeat), np.zeros(repeat)
            for i in range(repeat):
                costAll[i], trajTimeAll[i]  = self.runOneTrajectory(xy[0], xy[1])
                costU12[i] = self.tm.costU12
            meanCost = np.mean(costAll)
            meanTrajTime = np.mean(trajTimeAll)
            meanCostU12=np.mean(costU12)
            self.costStore.append([xy[0], xy[1], meanCost])
            self.trajTimeStore.append([xy[0], xy[1], meanTrajTime])
            self.cost12Store.append([xy[0], xy[1], meanCostU12])
            globMeanCost+=meanCost
            globTimeCost+=meanTrajTime
        #self.printLastCoordInfo()
        return globMeanCost/len(self.posIni), globTimeCost/len(self.posIni)
    
    def runTrajectoriesForResultsGenerationNController(self, repeat, thetaName):
        globMeanCost=0.
        globTimeCost=0.
        for enum,xy in enumerate(self.posIni):
            try :
                costAll, trajTimeAll, costU12 = np.zeros(repeat), np.zeros(repeat), np.zeros(repeat)
                controllerFileName = thetaName.replace("*",str(enum))
                self.tm.controller.load(controllerFileName)
                for i in range(repeat):
                    costAll[i], trajTimeAll[i]  = self.runOneTrajectory(xy[0], xy[1]) 
                    costU12[i] = self.tm.costU12
                meanCost = np.mean(costAll)
                meanTrajTime = np.mean(trajTimeAll)
                meanCostU12=np.mean(costU12)
                self.costStore.append([xy[0], xy[1], meanCost])
                self.trajTimeStore.append([xy[0], xy[1], meanTrajTime])
                self.cost12Store.append([xy[0], xy[1], meanCostU12])
                globMeanCost+=meanCost
                globTimeCost+=meanTrajTime
            except IOError:
                pass
        #self.printLastCoordInfo()
        return globMeanCost/len(self.posIni), globTimeCost/len(self.posIni)
    
    def runTrajectoriesForResultsGenerationOnePoint(self, repeat, point):
        xy = self.posIni[point]
        costAll, trajTimeAll = np.zeros(repeat), np.zeros(repeat)
        for i in range(repeat):
            costAll[i], trajTimeAll[i]  = self.runOneTrajectory(xy[0], xy[1]) 
        meanCost = np.mean(costAll)
        meanTrajTime = np.mean(trajTimeAll)
        return meanCost, meanTrajTime 
    
    def runTrajectoriesForResultsGenerationOpti(self, repeat):
        globMeanCost=0.
        globTimeCost=0.
        #pool=Pool()
        costAll, trajTimeAll = np.zeros(repeat), np.zeros(repeat)
        for xy in self.posIni:
            for i in range(repeat):
                costAll[i], trajTimeAll[i]  = self.runOneTrajectoryOpti(xy[0], xy[1]) 
            meanCost = np.mean(costAll)
            meanTrajTime = np.mean(trajTimeAll)
            self.costStore.append([xy[0], xy[1], meanCost])
            self.trajTimeStore.append([xy[0], xy[1], meanTrajTime])
            globMeanCost+=meanCost
            globTimeCost+=meanTrajTime
        #self.printLastCoordInfo()
        size=len(self.posIni)
        return globMeanCost/size, globTimeCost/size
    
    def runTrajectoriesForResultsGenerationEstim(self, repeat):
        globMeanCost=0.
        globTimeCost=0.
        #pool=Pool()
        costAll, trajTimeAll = np.zeros(repeat), np.zeros(repeat)
        for xy in self.posIni:
            for i in range(repeat):
                costAll[i], trajTimeAll[i]  = self.runOneTrajectoryEstim(xy[0], xy[1]) 
            meanCost = np.mean(costAll)
            meanTrajTime = np.mean(trajTimeAll)
            self.costStore.append([xy[0], xy[1], meanCost])
            self.trajTimeStore.append([xy[0], xy[1], meanTrajTime])
            globMeanCost+=meanCost
            globTimeCost+=meanTrajTime
        #self.printLastCoordInfo()
        size=len(self.posIni)
        return globMeanCost/size, globTimeCost/size
    
    def runMultiProcessTrajectories(self, repeat):
        pool=Pool(processes=len(self.posIni))
        result = pool.map(partial(self.runNtrajectory, repeat=repeat) , [(x, y) for x, y in self.posIni])
        pool.close()
        pool.join()
        meanCost, meanTraj=0., 0.
        for Cost, traj in result:
            meanCost+=Cost
            meanTraj+=traj
        size = len(result)
        return meanCost/size, meanTraj/size
       
    def runNtrajectory(self, (x, y), repeat):
        costAll, trajTimeAll = np.zeros(repeat), np.zeros(repeat)
        for i in range(repeat):
            costAll[i], trajTimeAll[i]  = self.runOneTrajectoryOpti(x, y) 
        meanCost = np.mean(costAll)
        meanTrajTime = np.mean(trajTimeAll)
        self.costStore.append([x, y, meanCost])
        self.trajTimeStore.append([x, y, meanTrajTime])
        return meanCost, meanTrajTime
    
    def mapableTrajecrtoryFunction(self,x,y,useless):
        return self.runOneTrajectory(x, y)
    
    def runNtrajectoryMulti(self, (x, y), repeat):
        pool=Pool(processes=4)
        result = pool.map(partial(self.mapableTrajecrtoryFunction,x,y) , range(repeat))
        pool.close()
        pool.join()
        meanCost, meanTraj=0., 0.
        for Cost, traj in result:
            meanCost+=Cost
            meanTraj+=traj
        size = len(result)
        return meanCost/size, meanTraj/size

    
    def runOneTrajectoryOpti(self, x, y):
        #self.tm.saveTraj = True
        cost, trajTime, lastX = self.tm.runTrajectoryOpti(x, y)
        #cost, trajTime, lastX = self.tm.runTrajectoryOpti(x, y)
        #print "Exp local x y cost : ", x, y, cost
        if lastX != -1000:
            self.lastCoord.append(lastX)
        return cost, trajTime
    
    def runOneTrajectoryEstim(self, x, y):
        #self.tm.saveTraj = True
        cost, trajTime, lastX = self.tm.runTrajectoryEstim(x, y)
        #cost, trajTime, lastX = self.tm.runTrajectoryOpti(x, y)
        #print "Exp local x y cost : ", x, y, cost
        if lastX != -1000:
            self.lastCoord.append(lastX)
        return cost, trajTime
    
    
    def runTrajectories(self,theta, fonction):
        '''
        Generates all the trajectories of the experimental setup and return the mean cost. This function is used by cmaes to optimize the controller.
    
        Input:        -theta: vector of parameters, one dimension normalized numpy array
    
        Ouput:        -meanAll: the mean of the cost of all trajectories generated, float
        '''
        
        #c = Chrono()
        self.initTheta(theta)
        #print "theta avant appel :", theta
        #compute all the trajectories x times each, x = numberOfRepeat
        meanCost, meanTime = fonction(self.numberOfRepeat)
        #cma.plot()
        #opt = cma.CMAOptions()
        #print "CMAES options :", opt
        #c.stop()

        #print("Indiv #: ", self.call, "\n Cost: ", meanCost)
        
        if (self.call==0):
            self.localBestCost = meanCost
            self.localWorstCost = meanCost
            self.localBestTime = meanTime
            self.localWorstTime = meanTime
            self.periodMeanCost = 0.0
            self.periodMeanTime = 0.0
        else:    
            if meanCost>self.localBestCost:
                self.localBestCost = meanCost
            elif meanCost<self.localWorstCost:
                self.localWorstCost = meanCost
                
            if meanTime>self.localBestTime:
                self.localBestTime = meanTime
            elif meanTime<self.localWorstTime:
                self.localWorstTime = meanTime

        if meanCost>self.bestCost:
            self.bestCost = meanCost
            fullfoldername = self.foldername+"Theta/"
            best_file_perf = find_best_theta_file(fullfoldername)
#            print('new max:',meanCost,'previous from file:',best_file_perf)
            if meanCost>0:
                if (best_file_perf<meanCost):
                    extension = ".save" + str(meanCost)
                    filename = findDataFilename(fullfoldername, "theta", extension)
                    np.savetxt(filename, self.theta)
                    filename2 = self.foldername + "Best.theta"
                    np.savetxt(filename2, self.theta)
#                else:
#                   print('A better controller already exists: ', best_file_perf,'>',meanCost)

        
        self.periodMeanCost += meanCost
        self.periodMeanTime += meanTime

        self.call += 1
        self.call = self.call%self.period

        if (self.call==0):
            self.periodMeanCost = self.periodMeanCost/self.period
            self.periodMeanTime = self.periodMeanTime/self.period
            self.CMAESCostStore.append((self.localWorstCost,self.periodMeanCost,self.localBestCost))
            self.CMAESTimeStore.append((self.localWorstTime,self.periodMeanTime,self.localBestTime))
            costfoldername = self.foldername+"Cost/"
            checkIfFolderExists(costfoldername)
            cost = open(costfoldername+"cmaesCost.log","a")
            time = open(costfoldername+"cmaesTime.log","a")
            cost.write(str(self.localWorstCost)+" "+str(self.periodMeanCost)+" "+str(self.localBestCost)+"\n")
            time.write(str(self.localWorstTime)+" "+str(self.periodMeanTime)+" "+str(self.localBestTime)+"\n")
            cost.close()
            time.close()
            #np.savetxt(costfoldername+"cmaesCost.log",self.CMAESCostStore) #Note: inefficient, should rather add to the file
            #np.savetxt(costfoldername+"cmaesTime.log",self.CMAESTimeStore) #Note: inefficient, should rather add to the file

        return 10.0*(self.rs.rhoCF-meanCost)/self.rs.rhoCF
    
    def runTrajectoriesCMAES(self, theta):
        '''
    	Generates all the trajectories of the experimental setup and return the mean cost. This function is used by cmaes to optimize the controller.
    
    	Input:		-theta: vector of parameters, one dimension normalized numpy array
    
    	Ouput:		-meanAll: the mean of the cost of all trajectories generated, float
    	'''
        return self.runTrajectories(theta, self.runMultiProcessTrajectories)
    

    def runTrajectoriesCMAESOnePoint(self, x, y, theta):
        '''
        Generates all the trajectories of the experimental setup and return the mean cost. This function is used by cmaes to optimize the controller.
    
        Input:        -theta: vector of parameters, one dimension normalized numpy array
    
        Ouput:        -meanAll: the mean of the cost of all trajectories generated, float
        '''
        return self.runTrajectories(theta, partial(self.runNtrajectory,(x,y)))

    def runTrajectoriesCMAESOnePointMulti(self, x, y, theta):
        return self.runTrajectories(theta, partial(self.runNtrajectoryMulti,(x,y)))
