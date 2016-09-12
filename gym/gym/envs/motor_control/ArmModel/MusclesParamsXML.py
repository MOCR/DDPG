#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Corentin Arnaud

Module: MusclesParameters

Description:    -We find here all muscles parameters
                
'''
import numpy as np
import math
import os 
from lxml import etree

class MusclesParameters:
    
    def __init__(self, nbJoint, nbMuscle):
        '''
        class parameters initialization
        '''
        self.pathSetupFile = os.getcwd() + "/ArmParams/MusclesParams"+str(nbJoint)+str(nbMuscle)+".xml"
        self.fmaxMatrix()
        
        ###############################Annexe parameters########################
        #Hogan parameters
        self.GammaMax = 2
        self.K = (2*self.GammaMax)/math.pi#stiffness
        self.Gamma_H = np.array([[0],[0]])#hogan torque initialization
        #stiffness matrix (null)
        self.Knulle = np.mat([(0, 0, 0, 0, 0, 0),(0, 0, 0, 0, 0, 0)])
        #stiffness matrix (low)
        self.Kp1 = 10.
        self.Kp2 = 10. 
        self.KP1 = 10.
        self.KP2 = 10.
        self.Kraid = np.mat([(self.KP1,self.KP1,0,0,self.Kp1,self.Kp1),(0,0,self.Kp2,self.Kp2,self.KP2,self.KP2)])   
        #stiffness matrix (high)
        self.KP22 = (80*self.GammaMax)/math.pi
        self.Kp22=(60*self.GammaMax)/math.pi
        self.KP11=(200*self.GammaMax)/math.pi
        self.Kp11=(100*self.GammaMax)/math.pi
        self.Kgrand = np.mat([(self.KP11,self.KP11,0,0,self.Kp11,self.Kp11),(0,0,self.Kp22,self.Kp22,self.KP22,self.KP22)])          
        #Proportional gain
        self.Kp = 10 # Arbitrary value       
        #Derivative gain
        self.Kd = 2*math.sqrt(self.Kp)    
        
    def fmaxMatrix(self):
        '''
        Defines the matrix of the maximum force exerted by each muscle
        '''
        tree = etree.parse(self.pathSetupFile).getroot()
        
        
        self.fmax = np.diag([int(fmaxi.text) for fmaxi in tree[0]])

        #line 7, amount of motor noise on U
        self.knoiseU = float(tree[1].text)
