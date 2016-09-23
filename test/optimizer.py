# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

"""

import gym
import numpy as np
import os

from Utils.FileWriting import checkIfFolderExists, findDataFilename, writeArray, find_best_theta_file, check_if_theta_file_exists
from Utils.ReadXmlArmFile import ReadXmlArmFile
from DDPG.core.DDPG_gym import DDPG_gym, save_DDPG, load_DDPG
from Experiments.StateEstimator import State_Estimator
from DDPG.core.helpers.read_xml_file import read_xml_file
from DDPG.test.helpers.logger import Logger

from gym.envs.classic_control import rendering
from gym.envs.motor_control.ArmModel.Arm import get_q_and_qdot_from

pathDataFolder = "./ArmParams/"
config = read_xml_file("DDPG_arm_config.xml")

controllers_folder = "./Ctrl/"

nb_repeats = 10
training_loops = 40

class Optimizer():
    def __init__(self):
        '''
    	Initializes parameters used to run functions below
     	'''
        self.rs  = ReadXmlArmFile(pathDataFolder + "setupArm.xml")
        self.env = gym.make('ArmModel-v0')
        self.state_estimator = State_Estimator(self.rs.inputDim, self.rs.outputDim, self.rs.delayUKF, self.env.get_arm())

    def reset(self):
        self.nb_steps = 0
        self.estim_state = self.state_estimator.init_store(self.env.reset())

    def step(self):
        action = np.array(self.learner.get_action_from_state(self.estim_state))
        delayed_state, reward, done, finished, _ = self.env._step(action)
        estim_next_state = self.state_estimator.get_estim_state(delayed_state,action)
        self.learner.store_sample(self.estim_state, action, reward, estim_next_state)
        if config.render:
            self.env.render()
        self.estim_state = estim_next_state
        q, qdot = get_q_and_qdot_from(self.estim_state)
        self.nb_steps += 1
        if config.render:
            self.add_estim_arm_view()
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
        viewer.add_onetime(arm_drawing)
        viewer.render(return_rgb_array = mode=='rgb_array')
    
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
            self.reset()
            total_cost, done = self.perform_repeats(nb_repeats, starting_point, target_size)
            if np.isnan(total_cost):
                print('cible',target_size,'episode',i,'divergence for point',starting_point)
                return best_cost
            self.logger.store(total_cost)
            if (total_cost>best_cost):
                if (total_cost>0):
                    foldername = controllers_folder + str(target_size) + "/" + str(starting_point) + "/"
                    self.save_controller(foldername,total_cost)
                best_cost=total_cost
                print('episode',i,'***** best cost',total_cost)
            self.learner.train_loop(training_loops)
        return best_cost
            
    def find_first_bests(self):
        finished = False
        while not finished:
            finished = True
            for target_size in [0.005, 0.01, 0.02, 0.04]:
                for i in range(15):
#                for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                    if not check_if_theta_file_exists(controllers_folder,target_size,i):
                        print('target',target_size,'point',i)
                        self.logger = Logger()
                        self.learner = DDPG_gym(self.env,config)
                        total = s.perform_M_episodes(100,i,target_size)
                        print('^^^^^^^^^^ final cost',total)
                        if total<=0:
                            finished = False
#                   self.logger.plot_progress()
            
    def optimize_existing(self):
        finished = False
        while not finished:
            for target_size in [0.005, 0.01, 0.02, 0.04]:
                for i in range(15):
                    print('target',target_size,'point',i)
                    self.logger = Logger()
                    foldername = controllers_folder + str(target_size) + "/" + str(i) + "/"
                    filename = foldername + "Best.theta"
                    self.learner = load_DDPG(self.env,filename)
                    total = s.perform_M_episodes(300,i,target_size)
                    print('^^^^^^^^^^ final cost',total)

    def save_controller(self,foldername,total_cost):
        filename1 = foldername + "Theta/theta1.save" + str(total_cost)
        filename2 = foldername + "Best.theta"
        fullfoldername = foldername+"Theta/"
        best_file_perf = find_best_theta_file(fullfoldername)
        if (best_file_perf<total_cost):
            checkIfFolderExists(fullfoldername)
#            print('saved in',filename1)
            save_DDPG(self.learner,filename1)
            save_DDPG(self.learner,filename2)

s = Optimizer()
s.find_first_bests()
s.optimize_existing()
