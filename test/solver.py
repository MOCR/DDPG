# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

"""

import gym
import numpy as np

from Utils.ReadXmlArmFile import ReadXmlArmFile
from DDPG.core.DDPG_gym import DDPG_gym
from Experiments.StateEstimator import State_Estimator
from DDPG.core.helpers.read_xml_file import read_xml_file

pathDataFolder = "./ArmParams/"
config = read_xml_file("DDPGconfig.xml")

#TODO: perform episode from each starting point (env.configure(i,target_size) env.reset())

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

    def step(self):
        action = np.array(self.learner.get_noisy_action_from_state(self.estim_state))
        delayed_state, reward, done, infos = self.env._step(action)
        estim_next_state = self.state_estimator.get_estim_state(delayed_state,action)
        self.learner.store_sample(self.estim_state, action, reward, estim_next_state)
        if config.render:
            self.env.render()
        vectarget = self.env.get_target_vector()
        self.estim_state = estim_next_state
        self.nb_steps += 1
        return reward, done
    
    def perform_episode(self):
        '''
        perform one episode
        numSteps is the number of steps over all episodes
        '''
        done = False
        while self.nb_steps < self.max_steps and not done:
            reward, done = self.step()
            self.learner.train_loop()
        return done

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
            done = self.perform_episode()
            if (self.nb_steps<max_nb_steps):
                max_nb_steps = self.nb_steps
                print('***** nb steps',self.nb_steps)
            else:
                print('nb steps',self.nb_steps)        

s = Solver()
for i in range(20):
    s.perform_M_episodes(15,0.04)
