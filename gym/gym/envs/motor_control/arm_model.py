# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

"""

import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

from gym.envs.motor_control.ArmModel.Arm import get_q_and_qdot_from
from gym.envs.motor_control.ArmModel.ArmType import ArmType
from gym.envs.motor_control.ArmModel.MuscularActivation import getNoisyCommand, muscleFilter
from gym.envs.motor_control.Cost.Cost import Cost

from Utils.ReadXmlArmFile import ReadXmlArmFile

pathDataFolder = "./ArmParams/"

class ArmModelEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    def __init__(self):
        '''
    	Initializes parameters used to run functions below
     	'''
        self.rs = ReadXmlArmFile(pathDataFolder + "setupArm.xml")
        self.name = "ArmModel"
        self.arm = ArmType[self.rs.arm]()
        self.arm.setDT(self.rs.dt)
        self.delay = self.rs.delayUKF
        self.eval = Cost(self.rs)

        if(not self.rs.det and self.rs.noise!=None):
            self.arm.setNoise(self.rs.noise)

        self.dimState = self.rs.inputDim
        self.dimOutput = self.rs.outputDim

        self.posIni = np.loadtxt(pathDataFolder + self.rs.experimentFilePosIni)
        if(len(self.posIni.shape)==1):
            self.posIni=self.posIni.reshape((1,self.posIni.shape[0]))
        
        self.max_speed = 5.0
        self.steps = 0
        self.t = 0
        
        self.min_action = np.zeros(6)
        self.max_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        low_pos = self.arm.armP.lowerBounds
        high_pos = self.arm.armP.upperBounds

        self.low_state = np.array(low_pos + [-self.max_speed, -self.max_speed])
        self.high_state = np.array(high_pos + [self.max_speed, self.max_speed])

        #Viewer init
        self.viewer = None
        self.screen_width = 600
        self.screen_height = 400
        world_width = 1.4
        self.scale = self.screen_width/world_width

        self.action_space = spaces.Box(self.min_action, self.max_action)
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self._seed()
        self.block_arm = False
#        self._configure(0,0.005)
#        self.arm.set_state(self.reset())

    def test_mgd(self):
        q1, q2 = self.arm.mgi(0.45,0.2)
        print ("x y : 0.45,0.2")
        print ("q1, q2 :",q1,q2)
        coordElbow, coordHand = self.arm.mgdFull([q1, q2])
        print ("coord reset", coordElbow, coordHand)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _configure(self, point_number=0, target_size=0.04):
        self.point_number = point_number
        self.target_size = target_size

    def _reset(self):
        q1, q2 = self.arm.mgi(self.posIni[self.point_number][0],self.posIni[self.point_number][1])
        self.state = [q1, q2, 0, 0]
        coordElbow, coordHand = self.arm.mgdFull([q1, q2])

        self.stateStore = np.zeros((self.delay,self.dimState))
        self.steps = 0
        self.t = 0
        self.block_arm = False
        return self.state
    
    def store_state(self, state):
        '''
    	Stores the current state and returns the delayed state
    
    	Input:		-state: the state to store
    	'''
        self.stateStore[1:]=self.stateStore[:-1]
        self.stateStore[0]=state
        return self.stateStore[self.delay-1]

    def get_current_state(self):
        return self.stateStore[0]

    def _step(self, action):

        Unoisy = getNoisyCommand(action,self.arm.musclesP.knoiseU)
        Unoisy = muscleFilter(Unoisy)
        #        Unoisy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if self.block_arm ==False:
            realNextState = self.arm.computeNextState(Unoisy, self.state)
        else:
            realNextState = self.state
        self.state = realNextState

        q, qdot = get_q_and_qdot_from(self.state)
        coordElbow, coordHand = self.arm.mgdFull(q)
        self.xy_elbow = coordElbow
        self.xy_hand = coordHand

        output_state = self.store_state(realNextState)
        cost, done, finished = self.eval.compute_reward(self.arm, self.t, Unoisy, self.steps, coordHand, self.target_size, self.get_current_state())
        self.steps += 1
        self.t += self.rs.dt
        
        if finished and not done and not self.block_arm and cost != 0:
            self.block_arm = True
            finished = False

        step_dic = {}
        step_dic['state'] = self.state
        step_dic['Unoisy'] = Unoisy
        step_dic['action'] = action
        step_dic['realNextState'] = realNextState
        step_dic['elbow'] = [coordElbow[0], coordElbow[1]]
        step_dic['hand'] = [coordHand[0], coordHand[1]]

        return realNextState, cost, done, finished, step_dic

    def scale_x(self,x):
        return self.screen_width/2 + x*self.scale

    def scale_y(self,y):
        return self.screen_height/4 + y*self.scale

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            
        xys = []
        xys.append([self.scale_x(0),self.scale_y(0)])
        x1 = self.scale_x(self.xy_elbow[0])
        y1 = self.scale_y(self.xy_elbow[1])
        xys.append([x1,y1])
        x2 = self.scale_x(self.xy_hand[0])
        y2 = self.scale_y(self.xy_hand[1])
        xys.append([x2,y2])
        
        arm_drawing = rendering.make_polyline(xys)
        arm_drawing.set_linewidth(4)
        arm_drawing.set_color(.8, .3, .3)
#       arm_drawing.add_attr(rendering.Transform())
        self.viewer.add_onetime(arm_drawing)

        xmin = self.rs.XTarget - self.target_size/2
        xmax = self.rs.XTarget + self.target_size/2
        ytarg = self.rs.YTarget 
        target = []
        top_line = []
        target.append([self.scale_x(xmin),self.scale_y(ytarg)])
        target.append([self.scale_x(xmax),self.scale_y(ytarg)])
        top_line.append([0,self.scale_y(ytarg)])
        top_line.append([self.screen_width,self.scale_y(ytarg)])
        
        bottom_line=[]
        bottom_line.append([0,self.scale_y(0.2)])
        bottom_line.append([self.screen_width,self.scale_y(0.2)])

        left_line=[]
        left_line.append([self.scale_x(-.3),0])
        left_line.append([self.scale_x(-.3),self.screen_height])
        
        right_line=[]
        right_line.append([self.scale_x(0.3),0])
        right_line.append([self.scale_x(0.3),self.screen_height])
        
        target_drawing = rendering.make_polyline(target)
        top_line_drawing = rendering.make_polyline(top_line)
        bottom_line_drawing = rendering.make_polyline(bottom_line)
        left_line_drawing = rendering.make_polyline(left_line)
        right_line_drawing = rendering.make_polyline(right_line)
        target_drawing.set_linewidth(4)
        target_drawing.set_color(.2, .3, .8)
        top_line_drawing.set_color(.9, .1, .1)
#        target_drawing.add_attr(rendering.Transform())
        self.viewer.add_geom(target_drawing)
        self.viewer.add_geom(top_line_drawing)
        self.viewer.add_geom(bottom_line_drawing)
        self.viewer.add_geom(left_line_drawing)
        self.viewer.add_geom(right_line_drawing)
        start1 = []
        start2 = []
        start3 = []
        for i in range(3):
            x_start = self.posIni[i][0]
            y_start = self.posIni[i][1]
            start1.append([self.scale_x(x_start),self.scale_y(y_start)])
        for i in range(5):
            x_start = self.posIni[i+3][0]
            y_start = self.posIni[i+3][1]
            start2.append([self.scale_x(x_start),self.scale_y(y_start)])
        for i in range(7):
            x_start = self.posIni[i+8][0]
            y_start = self.posIni[i+8][1]
            start3.append([self.scale_x(x_start),self.scale_y(y_start)])
        start1_drawing = rendering.make_polyline(start1)
        start1_drawing.set_color(.2, .8, .2)
        start2_drawing = rendering.make_polyline(start2)
        start2_drawing.set_color(.2, .8, .2)
        start3_drawing = rendering.make_polyline(start3)
        start3_drawing.set_color(.2, .8, .2)
#        start_drawing.add_attr(rendering.Transform())
        self.viewer.add_geom(start1_drawing)
        self.viewer.add_geom(start2_drawing)
        self.viewer.add_geom(start3_drawing)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_arm(self):
        return self.arm


    def get_viewer(self):
        return self.viewer

    def get_target_vector(self):
        qtarget1, qtarget2 = self.arm.mgi(self.rs.XTarget, self.rs.YTarget)
        return [qtarget1, qtarget2, 0.0, 0.0]

'''
a = ArmModelEnv()
a.test_mgd()
for i in range(20):
    a._step(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
'''
