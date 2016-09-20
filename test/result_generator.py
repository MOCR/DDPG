# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

"""

import gym
import numpy as np
import os

from Utils.FileWriting import checkIfFolderExists, findDataFilename, writeArray, find_best_theta_file
from Utils.ReadXmlArmFile import ReadXmlArmFile
from DDPG.core.DDPG_gym import DDPG_gym, load_DDPG
from Experiments.StateEstimator import State_Estimator
from DDPG.core.helpers.read_xml_file import read_xml_file
from DDPG.test.helpers.logger import Logger

from gym.envs.classic_control import rendering
from gym.envs.motor_control.ArmModel.Arm import get_q_and_qdot_from

pathDataFolder = "./ArmParams/"
config = read_xml_file("DDPG_arm_config.xml")
controllers_folder = "./Ctrl_save/"

class Result_Generator():
    def __init__(self):
        '''
    	Initializes parameters used to run functions below
     	'''
        self.rs  = ReadXmlArmFile(pathDataFolder + "setupArm.xml")
        self.env = gym.make('ArmModel-v0')
        self.state_estimator = State_Estimator(self.rs.inputDim, self.rs.outputDim, self.rs.delayUKF, self.env.get_arm())
        self.max_steps = self.rs.max_steps
        self.logger = Logger()
        self.posIni = np.loadtxt(pathDataFolder + self.rs.experimentFilePosIni)
        if(len(self.posIni.shape)==1):
            self.posIni=self.posIni.reshape((1,self.posIni.shape[0]))


    def reset(self):
        self.nb_steps = 0
        self.estim_state = self.state_estimator.init_store(self.env.reset())

    def store_step(self,dico):
        '''
        data [0 - 3] = target in joint space
        data [4 - 7] = estimated current state in joint space
        data [8 - 11] = actual current state in joint space
        data [12 - 17] = noisy muscular activations
        data [18 - 23] = noiseless muscular activations
        data [24 - 27] = estimated next state in joint space
        data [28 - 31] = actual next state in joint space
        data [32 - 33] = elbow position in cartesian space
        data [34 - 35] = hand position in cartesian space
        '''

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
        self.finalPoint = dico['hand']
        if (self.finalPoint[1]>=self.rs.YTarget):
            self.finalX.append(self.finalPoint[0])
        vec_save = np.array(vec_save).flatten()
        row = [item for sub in vec_save for item in sub]
        self.traj_store.append(row)

    def saveTarget(self):
        self.saveMeanTime()
        self.saveMeanCost()
        self.saveFinalX()
        
    def saveFinalX(self):
        foldername = self.rs.output_folder_name + str(self.ts) + "/finalX/"
        checkIfFolderExists(foldername)
        filename = foldername + "x.last"
        np.savetxt(filename,self.finalX)
        
    def saveMeanTime(self):
        foldername = self.rs.output_folder_name + str(self.ts) + "/TrajTime/"
        checkIfFolderExists(foldername)
        filename = foldername + "traj.time"
        np.savetxt(filename,self.time_store)
    
    def saveMeanCost(self):
        foldername = self.rs.output_folder_name + str(self.ts) + "/Cost/"
        checkIfFolderExists(foldername)
        filename = foldername + "traj.cost"
        np.savetxt(filename,self.cost_store)
        
    def saveTraj(self):
        foldername = self.rs.output_folder_name + str(self.ts) + "/Log/"
        checkIfFolderExists(foldername)
        filename = foldername + "traj" + str(self.cpt_traj) +  ".log"
        np.savetxt(filename,self.traj_store)

    def step(self):
        action = np.array(self.learner.get_action_from_state(self.estim_state))
        delayed_state, reward, done, finished, infos = self.env._step(action)
        estim_next_state = self.state_estimator.get_estim_state(delayed_state,action)
        self.learner.store_sample(self.estim_state, action, reward, estim_next_state)
        if config.render:
            self.env.render()
        infos['estimState'] = self.estim_state
        infos['vectarget'] = self.env.get_target_vector()
        infos['estimNextState'] = estim_next_state
        self.estim_state = estim_next_state
        q, qdot = get_q_and_qdot_from(self.estim_state)
        self.nb_steps += 1
        if not self.env.arm.is_inside_bounds(q):
            print ('estim q',q)
        if config.render:
            self.add_estim_arm_view()
        self.store_step(infos)
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
        self.traj_store = []

        t = 0
        total_cost = 0
        while not finished:
            reward, done, finished = self.step()
            total_cost += reward
            t += self.rs.dt
        self.saveTraj()
        self.cpt_traj += 1
        return total_cost, t, done

    def perform_M_episodes(self, M, starting_point, target_size):
        '''
        perform M episodes
        '''
        mean_cost = 0
        mean_time = 0
        done = False
        self.env.configure(starting_point, target_size)
        for i in range(M):
            self.reset()
            total_cost, total_time, done = self.perform_episode()
            self.logger.store(total_cost)
            mean_cost += total_cost
            mean_time += total_time
        return mean_cost/M, mean_time/M
            
    def generate_results(self,nb_repeats):
        for target_size in [0.005, 0.01, 0.02, 0.04]:
            self.cpt_traj = 0
            self.finalX = []
            self.cost_store = []
            self.time_store = []
            for i in range(15):
                print('target',target_size,'point',i)
                foldername = controllers_folder + str(target_size) + "/" + str(i) + "/"
                filename = foldername + "Best.theta"
                self.learner = load_DDPG(self.env,filename)
                self.learner.config.train = False
                self.ts = target_size
                self.pn = i
                mean_cost, mean_time = s.perform_M_episodes(nb_repeats,i,target_size)
                print('^^^^^^^^^^ mean cost', mean_cost)
                self.cost_store.append([self.posIni[i][0], self.posIni[i][1], mean_cost])
                self.time_store.append([self.posIni[i][0], self.posIni[i][1], mean_time])
            self.saveTarget()


s = Result_Generator()
s.generate_results(2)

