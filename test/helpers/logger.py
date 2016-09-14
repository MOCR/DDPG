# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

"""

import numpy as np
import os
import matplotlib.pyplot as plt

from Utils.ReadXmlArmFile import ReadXmlArmFile
from DDPG.core.helpers.read_xml_file import read_xml_file

class Logger():
    def __init__(self):
        '''
    	Initializes parameters used to run functions below
     	'''
        self.data_store = []

    def store(self,value):
        self.data_store.append(value)
        
    def plot_progress(self):
        plt.figure(1, figsize=(16,9))

        x,y = [],[]
        for j in range(len(self.data_store)):
            x.append(j)
            y.append(self.data_store[j])
        plt.plot(x, y, c = 'b')
 
        plt.title("Cost ")

        plt.savefig("costProgress.svg")
        plt.show(block = True)
