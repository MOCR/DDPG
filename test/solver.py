# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

"""

import gym
import numpy as np
import os

from Utils.ReadXmlArmFile import ReadXmlArmFile
from DDPG.core.DDPG_gym import DDPG_gym
from Experiments.StateEstimator import State_Estimator
from DDPG.core.helpers.read_xml_file import read_xml_file

pathDataFolder = "./ArmParams/"
config = read_xml_file("DDPGconfig.xml")

#TODO: perform episode from each starting point (env.configure(i,target_size) env.reset())

def checkIfFolderExists(name):
    if not os.path.isdir(name):
        os.makedirs(name)

def findDataFilename(foldername, name, extension):
    i = 1
    checkIfFolderExists(foldername)
    tryName = name + "1" + extension
    while tryName in os.listdir(foldername):
        i += 1
        tryName = name + str(i) + extension
    filename = foldername + tryName
    return filename

class Solver():
    def __init__(self):
        '''
    	Initializes parameters used to run functions below
     	'''
        self.rs  = ReadXmlArmFile(pathDataFolder + "setupArm.xml")
        self.env = gym.make('ArmModel-v0')
        self.learner = DDPG_gym(self.env,config)
        self.state_estimator = State_Estimator(self.rs.inputDim, self.rs.outputDim, self.rs.delayUKF, self.env.get_arm())
        self.estim_state = self.state_estimator.init_store(self.env.reset()) #TODO: improve: reset is done twice
        self.max_steps = self.rs.max_steps
        self.nb_steps = 0
        self.data_store = []

    def store_step(self,dico):
        vec_save = []
        vec_save.append(dico['vectarget'])
        vec_save.append(dico['estimState'])
        vec_save.append(dico['state'])
        vec_save.append(dico['Unoisy'])
        vec_save.append(dico['action'])
        vec_save.append(dico['estimNextState'])
        vec_save.append(dico['realNextState'])
        vec_save.append(dico['elbow'])
        vec_save.append(dico['hand'])
        vec_save = np.array(vec_save).flatten()
        row = [item for sub in vec_save for item in sub]
        self.data_store.append(row)

    def save(self):
        foldername = self.rs.output_folder_name
        filename = findDataFilename(foldername,"traj",".log")
        np.savetxt(filename,self.data_store)

    def step(self):
        action = np.array(self.learner.get_noisy_action_from_state(self.estim_state))
        delayed_state, reward, done, finished, infos = self.env._step(action)
        estim_next_state = self.state_estimator.get_estim_state(delayed_state,action)
        self.learner.store_sample(self.estim_state, action, reward, estim_next_state)
        if config.render:
            self.env.render()
        infos['estimState'] = self.estim_state
        infos['vectarget'] = self.env.get_target_vector()
        infos['estimNextState'] = estim_next_state
        self.estim_state = estim_next_state
        self.nb_steps += 1
        self.store_step(infos)
        return reward, done, finished
    
    def perform_episode(self):
        '''
        perform one episode
        numSteps is the number of steps over all episodes
        '''
        finished = False
        total_cost = 0
        while not finished:
            reward, done, finished = self.step()
            total_cost += reward
            self.learner.train_loop()
        self.save()
        return total_cost, done

    def perform_M_episodes(self, M, target_size):
        '''
        perform M episodes
        '''
        max_nb_steps = 10000 #OSD:patch to study nb steps
        done = False
        for i in range(M):
            self.env.configure(i%15, target_size)
            self.nb_steps = 0
            self.state = self.env.reset()
            total_cost, done = self.perform_episode()
            print('total cost',total_cost)
            if (self.nb_steps<max_nb_steps):
                max_nb_steps = self.nb_steps
                print('***** nb steps',self.nb_steps)
            else:
                print('nb steps',self.nb_steps)        

s = Solver()
for i in range(20):
    s.perform_M_episodes(15,0.04)
