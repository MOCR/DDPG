#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Corentin Arnaud

Module: ReadXmlFile

Description: reads a global xml setup file
'''

from lxml import etree
from os import path
import math

class ReadXmlArmFile(object):
    def __init__(self, xmlFile):
        self.parse(xmlFile)

    def parse(self, xmlFile):
        tree = etree.parse(xmlFile).getroot()
        self.dataParse(tree[0])
        self.costFunctionParse(tree[1])
        self.targetParse(tree[2])
        self.trajectoryParse(tree[3])
        self.kalmanParse(tree[4])
        self.outputFolderParse(tree[5])
    
    def dataParse(self, dataElement):
        self.inputDim=int(dataElement[0].text)
        self.outputDim=int(dataElement[1].text)
        self.det="no"==dataElement[2].text
        if(not self.det and dataElement[2].text!= 'yes'):
            self.noise = float(dataElement[2].text)
        else :
            self.noise=None
        self.arm="Arm"+str(self.inputDim/2)+str(self.outputDim)
        
    def costFunctionParse(self, cf):
        self.gammaCF=float(cf[0].text)
        self.rhoCF  =float(cf[1].text)
        self.upsCF  =float(cf[2].text)
        
    def targetParse(self, targetElement):
        self.target_size = []
        for size in targetElement[0]:
            self.target_size.append(float(size.text))
        self.XTarget = float(targetElement[1][0].text)
        self.YTarget = float(targetElement[1][1].text)
        
    def trajectoryParse(self, trajectoryElement):
        self.experimentFilePosIni=trajectoryElement[0].text
        self.max_steps=int(trajectoryElement[1].text)
        self.dt=float(trajectoryElement[2].text)
      
    def getDistanceToTarget(self, x, y):
        return math.sqrt((x - self.XTarget)**2 + (y - self.YTarget)**2) 
    
        
    def kalmanParse(self, kalmanElement):
        self.delayUKF=int(kalmanElement[0].text)
        
    def outputFolderParse(self, output):
        self.output_folder_name=output[0].text
