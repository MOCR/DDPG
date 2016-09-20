#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Thomas Beucher, Corentin Arnaud
Module: FileReading
Description: Functions to read project data

Organisation of the file :

data [0 - 3] = target in joint space
data [4 - 7] = estimatedcurrent state in joint space
data [8 - 11] = actual current state in joint space
data [12 - 17] = noisy muscular activations
data [18 - 23] = noiseless muscular activations
data [24 - 27] = estimated next state in joint space
data [28 - 31] = actual next state in joint space
data [32 - 33] = elbow position in cartesian space
data [34 - 35] = hand position in cartesian space
'''
import random as rd
import numpy as np
import os
import glob

from ArmModel.ArmType import ArmType

def loadTrajs(folderName, prct, det=False):
    '''
    Get all the data from a set of trajectories, sorted by the starting xy coordinates
    
    Output :               -state: np-array of trajectory's states
                           -activity: np-array of trajectory's activity
    '''
    listdir = os.listdir(folderName)
    state=[]
    activity=[]
    nbTraj = 0
    for trajFile in listdir:
        if(rd.random() < prct):
            tmpData = np.loadtxt(folderName + trajFile)
            stateTrajectory    = np.empty((tmpData.shape[0],4))
            activityTrajectory = np.empty((tmpData.shape[0],6))
            for i in range(tmpData.shape[0]):
                stateTrajectory[i] = tmpData[i][8:12]
                #wiht noise
                if(det==False):
                    activityTrajectory[i] = tmpData[i][12:18]
                else :
                    activityTrajectory[i] = tmpData[i][18:24]
            state.append(stateTrajectory)
            activity.append(activityTrajectory)
            nbTraj+=1
    print(str(nbTraj) + " Trajectory charged")
    return np.array(state), np.array(activity)

def loadExpeTrajs(folderName, prct):
    '''
    Get all the data from a subject, sorted by the starting xy coordinates
    
    Output :               -state: np-array of trajectory's states
                           -activity: np-array of trajectory's activity
    '''
    listdir = glob.glob(folderName+"/*")
    coor=[]
    velocity=[]
    time=[]
    pos=[]
    nbTraj = 0
    for trajFile in listdir:
        if(rd.random() < prct):
            tmpData = np.loadtxt(trajFile)
            coorTrajectory    = np.empty((tmpData.shape[0],2))
            velocityTrajectory = np.empty((tmpData.shape[0],2))
            timeTrajectory= np.empty((tmpData.shape[0]))
            if(timeTrajectory[-1]>4) : continue
            for i in range(tmpData.shape[0]):
                timeTrajectory[i]=tmpData[i][0]
                coorTrajectory[i] = tmpData[i][1:3]
                velocityTrajectory[i] = tmpData[i][3:]
            coor.append(coorTrajectory)
            velocity.append(velocityTrajectory)
            time.append(timeTrajectory)
            pos.append(int(os.path.basename(trajFile)[0]))                   #TODO: extract filename[0]

            nbTraj+=1
    print(str(nbTraj) + " Trajectory charged")
    return np.array(time), np.array(coor), np.array(velocity), np.array(pos)

    

#TODO: Warning only for Arm26
def loadStateCommandPairsByStartCoords(foldername, prct, det=False):
    '''
    Get all the data from a set of trajectories, sorted by the starting xy coordinates
    Work only for a 2 join 6 muscle arm
    Output : dictionary of data whose keys are y then x coordinates
    '''
    arm = ArmType["Arm26"]()
    dataOut = {}
    listdir = os.listdir(foldername)
    for el in listdir:
#        j = j+1
#        if j>4500 or rd.random()<0.5:
        if(rd.random() < prct):
            data = np.loadtxt(foldername + el)
            coordHand = arm.mgdEndEffector(np.array([[data[0,10]], [data[0,11]]]))
            x,y = str(coordHand[0][0]), str(coordHand[1][0])
            if not y in dataOut.keys():
                dataOut[y] = {}
            if not x in dataOut[y].keys():
                dataOut[y][x] = []
            traj = []
            for i in range(data.shape[0]):
                currentState = (data[i][8], data[i][9], data[i][10], data[i][11])
                #wiht noise
                if(det==False):
                    Activ = ([data[i][12], data[i][13], data[i][14], data[i][15], data[i][16], data[i][17]])
                else :
                    Activ = ([data[i][18], data[i][19], data[i][20], data[i][21], data[i][22], data[i][23]])
                pair = (currentState, Activ)
                traj.append(pair)
            dataOut[y][x].append(traj)
    return dataOut

def stateAndCommandDataFromTrajs(data):
        '''
        Reorganizes trajectory data into an array of trajectories made of (state, command) pairs
        
        Input:     -data: dictionary
        Output:    -dataA: numpy array
        '''
        state, command = [], []
        for key, _ in data.items():
            #if float(key) < 0.58:
                for _, xvals in data[key].items():
                    for i in range(len(xvals)):
                        stateVec, commandVec = [], []
                        for j in range(len(xvals[i])):
                            stateVec.append(xvals[i][j][0])
                            commandVec.append(xvals[i][j][1])
                        state.append(stateVec)
                        command.append(commandVec)
        '''
        state = np.vstack(np.array(state))
        command = np.vstack(np.array(command))
        '''
        return state,command

# -------------------------------------------------------------------------------------------
    
#TODO: Warning only for Arm26
def getInitPos(foldername):
    '''
    Get all the initial positions from a set of trajectories, in xy coordinates
    
    Output : dictionary of initial position of all trajectories
    '''
    arm = ArmType["Arm26"]()
    xy = {}
    for el in os.listdir(foldername):
            data = np.loadtxt(foldername + el)
            coordHand = arm.mgdEndEffector(np.array([[data[0,10]], [data[0,11]]]))
            #if coordHand[1]<0.58:
            xy[el] = (coordHand[0], coordHand[1])
    return xy
  
def getStateAndCommandData(foldername):
    '''
    Put all the states and commands of trajectories generated by the Brent controller into 2 different dictionaries
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
                -command: dictionary keys = filenames, values = array of data
    '''
    state, command = {}, {}
    for el in os.listdir(foldername):
        #if rd.random()<0.06:
            state[el], command[el] = [], []
            data = np.loadtxt(foldername + el)
            for i in range(data.shape[0]):
                state[el].append((data[i][8], data[i][9], data[i][10], data[i][11]))
                #command[el].append((data[i][18], data[i][19], data[i][20], data[i][21], data[i][22], data[i][23]))
                #we use the noisy command because it is filtered
                #com = np.array([data[i][12], data[i][13], data[i][14], data[i][15], data[i][16], data[i][17]])
                #com = muscleFilter(com)
                #It seems that filtering prevents learning...
                command[el].append([data[i][12], data[i][13], data[i][14], data[i][15], data[i][16], data[i][17]])
    return state, command
  
def getCommandData(foldername):
    '''
    Put all the states and commands of trajectories generated by the Brent controller into 2 different dictionaries
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    -command: dictionary keys = filenames, values = array of data
    '''
    command = {}
    for el in os.listdir(foldername):
            command[el] = []
            data = np.loadtxt(foldername + el)
            for i in range(data.shape[0]):
                #command[el].append((data[i][18], data[i][19], data[i][20], data[i][21], data[i][22], data[i][23]))
                #we use the noisy command because it is filtered
                command[el].append([data[i][12], data[i][13], data[i][14], data[i][15], data[i][16], data[i][17]])
    return command
  
def getNoiselessCommandData(foldername):
    '''
    Put all the states and commands of trajectories generated by the Brent controller into 2 different dictionaries
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    -command: dictionary keys = filenames, values = array of data
    '''
    command = {}
    for el in os.listdir(foldername):
            command[el] = []
            data = np.loadtxt(foldername + el)
            for i in range(data.shape[0]):
                command[el].append((data[i][18], data[i][19], data[i][20], data[i][21], data[i][22], data[i][23]))
                #we use the noisy command because it is filtered
                #command[el].append([data[i][12], data[i][13], data[i][14], data[i][15], data[i][16], data[i][17]])
    return command

def getStateData(foldername):
    '''
    Put all the states of trajectories generated by the Brent controller into a dictionary
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    '''
    state = {}
    for el in os.listdir(foldername):
        state[el] = []
        data = np.loadtxt(foldername + el)
        for i in range(data.shape[0]):
            state[el].append((data[i][8], data[i][9], data[i][10], data[i][11]))
    return state

def getXYHandData(foldername, factor=0):
    '''
    Put all the states of trajectories into a dictionary
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    '''
    dir = os.listdir(foldername)
    if(factor != 0): factor = min(1, float(factor)/len(dir))
    arm = ArmType["Arm26"]()
    xy = {}
    for el in dir:
        if  rd.random()<factor:
            xy[el] = []
            data = np.loadtxt(foldername + el)
            for i in range(data.shape[0]):
                coordHand = arm.mgdEndEffector(np.array([[data[i][8]], [data[i][9]]]))
                xy[el].append((coordHand[0], coordHand[1]))
    return xy

def getXYElbowData(foldername):
    '''
    Put all the states of trajectories generated by the Brent controller into a dictionary
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    '''
    arm = ArmType["Arm26"]()
    xy = {}
    for el in os.listdir(foldername):
        xy[el] = []
        data = np.loadtxt(foldername + el)
        for i in range(data.shape[0]):
            coordElbow, _ = arm.mgdFull(np.array([[data[i][8]], [data[i][9]]]))
            xy[el].append((coordElbow[0], coordElbow[1]))
    return xy

def getEstimatedStateData(foldername):
    '''
    Put all the states of trajectories generated by the Brent controller into a dictionary
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    '''
    state = {}
    for el in os.listdir(foldername):
        state[el] = []
        data = np.loadtxt(foldername + el)
        for i in range(data.shape[0]):
            state[el].append((data[i][4], data[i][5], data[i][6], data[i][7]))
    return state
    
def getEstimatedXYHandData(foldername):
    '''
    Put all the states of trajectories generated by the Brent controller into a dictionary
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    '''
    arm = ArmType["Arm26"]()
    xyEstim = {}
    for el in os.listdir(foldername):
        xyEstim[el] = []
        data = np.loadtxt(foldername + el)
        for i in range(data.shape[0]):
            coordHand = arm.mgdEndEffector(np.array([[data[i][4]], [data[i][5]]]))
            xyEstim[el].append((coordHand[0], coordHand[1]))
    return xyEstim

#TODO: Warning only for Arm26   
def getXYEstimError(foldername):
    '''
    Returns the error estimations in the trajectories from the given foldername
    
    Outputs:    -errors: dictionary keys = filenames, values = array of data
    '''
    arm = ArmType["Arm26"]()
    errors = {}
    for el in os.listdir(foldername):
        errors[el] = []
        data = np.loadtxt(foldername + el)
        for i in range(data.shape[0]):
            statePos = (data[i][8], data[i][9])
            estimStatePos = (data[i][4], data[i][5])
            errors[el].append(arm.estimErrorReduced(statePos,estimStatePos))
    return errors

#TODO: Warning only for Arm26    
def getXYEstimErrorOfSpeed(foldername):
    '''
    Returns the error estimations in the trajectories as a function of speed from the given foldername
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    '''
    arm = ArmType["Arm26"]()
    errors = {}
    for el in os.listdir(foldername):
        errors[el] = []
        data = np.loadtxt(foldername + el)
        for i in range(data.shape[0]):
            speed = arm.cartesianSpeed((data[i][8], data[i][9], data[i][10], data[i][11]))
            statePos = (data[i][8], data[i][9])
            estimStatePos = (data[i][4], data[i][5])
            error = arm.estimErrorReduced(statePos,estimStatePos)
            errors[el].append((speed, error))
    return errors
    
def getCostData(foldername):
    '''
    Put all the states of trajectories generated by the Brent controller into a dictionary
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    '''
    costDico = {}
    for el in os.listdir(foldername):
        costDico[el] = []
        data = np.loadtxt(foldername + el)
        for i in range(data.shape[0]):
            x = data[i][0]
            y = data[i][1]
            cost = data[i][2]
            costDico[el].append((x, y, cost))
    return costDico
    
def getTrajTimeData(foldername):
    '''
    Put all the states of trajectories generated by the Brent controller into a dictionary
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    '''
    trajTimeDico = {}
    for el in os.listdir(foldername):
        trajTimeDico[el] = []
        data = np.loadtxt(foldername + el)
        for i in range(data.shape[0]):
            x = data[i][0]
            y = data[i][1]
            trajTime = data[i][2]
            trajTimeDico[el].append((x, y, trajTime))
    return trajTimeDico
    
def getLastXData(foldername):
    '''
    Put all the states of trajectories generated by the Brent controller into a dictionary
    
    Outputs:    -state: dictionary keys = filenames, values = array of data
    '''
    xDico = {}
    for el in os.listdir(foldername):
        xDico[el] = []
        data = np.loadtxt(foldername + el)
        for i in range(data.shape[0]):
            xDico[el].append(data[i])
    return xDico

def dicToArray(data):
        '''
        This function transform a dictionary into an array
        
        Input:     -data: dictionary
        Output:    -dataA: numpy array
        '''
        retour = []
        for _, v in data.items():
            retour.append(v)
        return np.vstack(np.array(retour))
            
    

def loadTrajForModel(folderName, delay):
    '''
    
    Input :                -foldername : name of the folder that contain trajectory
                           -delay : Kalman delay
    
    Output :               -state: np-array of trajectory's states
                           -activity: np-array of trajectory's activity
    '''
    listdir = os.listdir(folderName)
    stateAndCommand=[]
    nextState=[]
    nbTraj = 0
    for trajFile in listdir:
        tmpData = np.loadtxt(folderName + trajFile)
        stateAndCommandtmp    = np.empty((tmpData.shape[0]-delay,4+6*delay))
        nextStatetmp = np.empty((tmpData.shape[0]-delay,4))
        for i in range(tmpData.shape[0]-delay):
            stateAndCommandtmp[i][:4] = tmpData[i][8:12]
            for j in range(delay):
                stateAndCommandtmp[i][4+6*j:4+6*(j+1)]=tmpData[i+j][18:24]
            nextStatetmp[i]=tmpData[i+j+1][8:12]
        stateAndCommand.append(stateAndCommandtmp)
        nextState.append(nextStatetmp)
        nbTraj+=1
    print(str(nbTraj) + " Trajectory charged")
    return np.vstack(stateAndCommand), np.vstack(nextState)
    



