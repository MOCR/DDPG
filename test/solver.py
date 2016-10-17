# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

"""

import gym
import numpy as np
import os
import math

from Utils.ReadXmlArmFile import ReadXmlArmFile
from DDPG.core.DDPG_gym import DDPG_gym
from Experiments.StateEstimator import State_Estimator
from DDPG.core.helpers.read_xml_file import read_xml_file
from DDPG.test.helpers.logger import Logger

import matplotlib.pyplot as plt

from gym.envs.classic_control import rendering
from gym.envs.motor_control.ArmModel.Arm import get_q_and_qdot_from

import random

pathDataFolder = "./ArmParams/"
config = read_xml_file("DDPG_arm_config.xml")

nb_repeats = 1

#TODO: perform episode from each starting point (env.configure(i,target_size) env.reset())

def dist_vects(v1,v2):
    dist = 0
    for i in range(len(v1)):
        dist+= math.pow(v1[i]-v2[i],2.0)
    return math.sqrt(dist)

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
        self.max_steps = self.rs.max_steps
        self.data_store = []
        self.logger = Logger()
        self.action_curve=[]

    def reset(self):
        self.nb_steps = 0
        #state = []
        #for i in range(self.rs.inputDim*2):
        #    state.append(0.0)
        #self.estim_state = self.state_estimator.init_store(self.env.reset())
        #for i in range(len(self.estim_state)):
        #    state.append(self.estim_state[i])
        self.estim_state = self.env.reset()
            

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
        #pract = action
        #print action.shape
        #action = np.array([max(action[0],0), -min(action[0],0),max(action[1],0), -min(action[1],0),max(action[2],0), -min(action[2],0)])     
        #action = np.array(self.learner.get_action_from_state(self.estim_state))
        delayed_state, reward, done, finished, infos = self.env._step(action)
        #reward = reward/1000.0
        self.action_curve.append(action)
        #self.action_curve.append(self.learner.value([[self.estim_state], [action]])[0])
        #state = self.estim_state
        #estim_next_state = self.state_estimator.get_estim_state(delayed_state,action)
        #for i in range(len(estim_neif dist_vects(delayed_state, self.estim_state)<0.1:
        #    state.pop(0)
        #    state.append(estim_next_state[i])
        #estim_next_state = state
        estim_next_state = delayed_state
        self.learner.store_sample(self.estim_state, action, reward, estim_next_state)
        if config.render:
            self.env.render()
#        infos['estimState'] = self.estim_state
#        infos['vectarget'] = self.env.get_target_vector()
#        infos['estimNextState'] = estim_next_state
        if dist_vects(estim_next_state, self.estim_state)<0.01:
            reward-=10
        self.estim_state = estim_next_state
        q, qdot = get_q_and_qdot_from(self.estim_state)
        self.nb_steps += 1
        if not self.env.arm.is_inside_bounds(q):
            print ('estim q',q)
        if config.render:
            #print self.estim_state
            self.add_estim_arm_view()
        self.learner.train()
#        self.store_step(infos)
        return reward, done, finished

    def add_estim_arm_view(self, mode='human'):
        viewer = self.env.get_viewer()
        q, qdot = get_q_and_qdot_from(self.estim_state)
        xy_elbow, xy_hand = self.env.arm.mgdFull(q)
        xys = []
        xys.append([self.env.scale_x(0),self.env.scale_y(0)])
        x1 = self.env.scale_x(xy_elbow[0])
        y1 = self.env.scale_y(xy_elbow[1])
        xys.append([x1,y1])
        x2 = self.env.scale_x(xy_hand[0])
        y2 = self.env.scale_y(xy_hand[1])
        xys.append([x2,y2])
        
        arm_drawing = rendering.make_polyline(xys)
        arm_drawing.set_linewidth(4)
        arm_drawing.set_color(.1, .1, .8)
#       arm_drawing.add_attr(rendering.Transform())
        viewer.add_onetime(arm_drawing)
        #viewer.render(return_rgb_array = mode=='rgb_array')
    
    def perform_episode(self):
        '''
        perform one episode
        numSteps is the number of steps over all episodes
        '''
        finished = False
        total_cost = 0
        self.env.configure(random.uniform(0,14), 0.04)
        while not finished:
            reward, done, finished = self.step()
            total_cost += reward
#        self.save()
        #plt.plot(self.reward_curve)
        #plt.show()
        return total_cost, done

    def perform_repeats(self, repeats, starting_point, target_size):
        mean_cost = 0
        done = False
        for i in range(repeats):
            self.reset()
            total_cost, done = self.perform_episode()
            mean_cost += total_cost
        return mean_cost/repeats, done
    
    def perform_M_episodes(self, M, starting_point, target_size):
        '''
        perform M episodes
        '''
        best_cost = -30000
        done = False
        self.env.configure(starting_point, target_size)
        for i in range(M):
            #self.learner.noise_generator.randomRange()
            #self.env.configure(i%15, target_size)
            if i%10==0:
                config.render=True
            else:
                config.render=False
            self.reset()
            total_cost, done = self.perform_repeats(nb_repeats, starting_point, target_size)
            self.logger.store(total_cost)
            plt.close()
            plt.plot(self.action_curve)
            plt.show(block=False)
            if (total_cost>best_cost):
                best_cost=total_cost
                print('episode',i,'***** best cost',total_cost)
            else:
                print('episode',i,'total cost',total_cost)
            #self.learner.train_loop(25)
        return best_cost

s = Solver()
for ii in range(100):
    for i in range(15):
        total = s.perform_M_episodes(10000,i,0.04)
        print('^^^^^^^^^^ final cost',total)
        s.logger.plot_progress()
