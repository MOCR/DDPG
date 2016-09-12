#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Corentin Arnaud

Module: ArmParameters

Description:    -We find here all arm parameters

'''
import numpy as np
import os
from lxml import etree

class ArmParameters:
    '''
    class ArmParameters
    '''
    
    def __init__(self, nbJoint, nbMuscle):
        '''
        Intializes the class
        '''
        self.pathSetupFile = os.getcwd() + "/ArmParams/ArmParams"+str(nbJoint)+str(nbMuscle)+".xml"
        tree = etree.parse(self.pathSetupFile).getroot()
        self.nbJoint=nbJoint
        self.nbMuscle=nbMuscle
        
        self.readSetup(tree[0])
        self.BMatrix(tree[1])
        self.AMatrix(tree[2])

        self.readStops(tree[3])
        
    def readSetup(self,tree):
        '''
        Reads the setup file
        '''
        self.l=np.empty(self.nbJoint)
        self.m=np.empty(self.nbJoint)
        self.I=np.empty(self.nbJoint)
        self.s=np.empty(self.nbJoint)
        
        for i, part in enumerate(tree[0]):
            self.l[i]=float(part.text)
        for i, part in enumerate(tree[1]):
            self.m[i]=float(part.text)
        for i, part in enumerate(tree[2]):
            self.I[i]=float(part.text)
        for i, part in enumerate(tree[3]):
            self.s[i]=float(part.text)   
            

    
    def BMatrix(self,tree):
        '''
        Defines the damping matrix B
        '''

        self.B = np.array([float(b.text) for b in tree ]).reshape((self.nbJoint,self.nbJoint))
    
    def AMatrix(self,tree):
        '''
        Defines the moment arm matrix A
        '''

        self.At = np.array([float(a.text) for a in tree]).reshape((self.nbJoint,self.nbMuscle))

    def readStops(self,tree):
        self.lowerBounds=[]
        self.upperBounds=[]
        for i, parts in enumerate(tree):
            self.lowerBounds.append(float(parts[0].text))
            self.upperBounds.append(float(parts[1].text))
            

    
    
    
