from Utils.FileReading import loadTrajForModel
from GlobalVariables import  pathDataFolder
import numpy as np

import os
from Regression.NeuralNet import NeuralNet
import sys

#for avoid circular import problem 
def typeImport():
    from Regression.RunRegression import regressionDict













def run(regressionSetup, delay):
    
    
    stateAndCommand, nextState = loadTrajForModel(pathDataFolder + "Brent/", delay)

    
    
    
    print("nombre d'echantillons: ", len(stateAndCommand))

    fa = regressionDict[regressionSetup.regression](regressionSetup)
    fa.getTrainingData(stateAndCommand, nextState)
    fa.train()
    
    
class NeuraNetParameter():
        def __init__(self, delay, name):
            self.regression=name
            self.thetaFile="EstimThetaNeuralNetNpTF"+str(delay)
            self.path = os.getcwd()+"/Experiments/theta/"
            self.inputLayer="linear"
            self.outputLayer="linear"
            self.hiddenLayers=[]
            for i in range(3):
                self.hiddenLayers.append(("tanh",100))
            self.inputDim=4+6*delay
            self.outputDim=4
            self.learningRate=0.001
            self.momentum=0.
            self.bias=True
            
            
            
if __name__ == "__main__" :
    
    delay = int(sys.argv[1])
    regressionSetup = NeuraNetParameter(delay, "NeuralNetTF")
    run(regressionSetup, delay)