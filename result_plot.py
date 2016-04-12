# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:06:10 2016

@author: debroissia
"""

#import time
#import numpy as np
import matplotlib.pyplot as plt

class result_plot:
    def __init__(self):
        #plt.axis([0, 1000, 0, 1])
        plt.ion()
        plt.show()
        self.i = 0
    
    def add_data(self, y):
        plt.scatter(self.i, y)
        self.i += 1
        plt.draw()
    def add_row(self, Y):
#        for y in Y:
#            plt.scatter(self.i, y)
#            self.i += 1
        plt.plot(Y, figure=plt.gcf())
        plt.draw()
    def clear(self):
        plt.cla()