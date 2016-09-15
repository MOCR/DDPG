#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Corentin Arnaud

Module: Arm

Description:    
-Models an arm with two joints and six muscles
-Computes its dynamics
'''

import numpy as np
import math

from gym.envs.motor_control.ArmModel.Arm import *
from gym.envs.motor_control.ArmModel.ArmParamsXML import ArmParameters
from gym.envs.motor_control.ArmModel.MusclesParamsXML import MusclesParameters

#-----------------------------------------------------------------------------

class Arm26(Arm):
    def __init__(self):
        Arm.__init__(self, ArmParameters(2,6),MusclesParameters(2,6),  np.array([0.,0.]))
        self.k = np.array([self.armP.I[0] + self.armP.I[1] + self.armP.m[1]*(self.armP.l[1]**2), self.armP.m[1]*self.armP.l[0]*self.armP.s[1], self.armP.I[1]])

    def computeNextState(self, U, state):
        '''
        Computes the next state resulting from the direct dynamic model of the arm given the muscles activation vector U
    
        Inputs:     -U: (6,1) numpy array
        -state: (4,1) numpy array (used for Kalman, not based on the current system state)

        Output:    -state: (4,1) numpy array, the resulting state
        '''
        #print ("state:", state)
        q, qdot = get_q_and_qdot_from(state)
        #print ("U :",U)
        #print ("qdot:",qdot)
        M = np.array([[self.k[0]+2*self.k[1]*math.cos(q[1]),
                       self.k[2]+self.k[1]*math.cos(q[1])],
                      [self.k[2]+self.k[1]*math.cos(q[1]),
                       self.k[2]]])
        #print ("M:",M)
        #Minv = np.linalg.inv(M)
        #print ("Minv:",Minv)
        C = np.array([-qdot[1]*(2*qdot[0]+qdot[1])*self.k[1]*math.sin(q[1]),(qdot[0]**2)*self.k[1]*math.sin(q[1])])
        #print ("C:",C)
        #the commented version uses a non null stiffness for the muscles
        #beware of dot product Kraid times q: q may not be the correct vector/matrix
        #Gamma = np.dot((np.dot(armP.At, musclesP.fmax)-np.dot(musclesP.Kraid, q)), U)
        #Gamma = np.dot((np.dot(self.armP.At, self.musclesP.fmax)-np.dot(self.musclesP.Knulle, Q)), U)
        #above Knulle is null, so it can be simplified

        Gamma = np.dot(np.dot(self.armP.At, self.musclesP.fmax), U)
        #print ("Gamma:",Gamma)

        #Gamma = np.dot(armP.At, np.dot(musclesP.fmax,U))
        #computes the acceleration qddot and integrates
    
        b = np.dot(self.armP.B, qdot)
        #print ("b:",b)

        #qddot = np.dot(Minv,Gamma - C - b)
        #print ("qddot",qddot)

        #To avoid inverting M:
        qddot = np.linalg.solve(M, Gamma - C - b)
    
        qdot += qddot*self.dt
        q += qdot*self.dt
        #save the real state to compute the state at the next step with the real previous state
        q = self.joint_stop(q)
        nextState = np.array([q[0], q[1], qdot[0], qdot[1]])
        return nextState

    
    def mgdFull(self, q):
        '''
        Direct geometric model of the arm
    
        Inputs:     -q: (2,1) numpy array, the joint coordinates
    
        Outputs:
        -coordElbow: elbow coordinate
        -coordHand: hand coordinate
        '''
        coordElbow = [self.armP.l[0]*np.cos(q[0]), self.armP.l[0]*np.sin(q[0])]
        coordHand = [self.armP.l[0]*np.cos(q[0])+self.armP.l[1]*np.cos(q[0] + q[1]), self.armP.l[0]*np.sin(q[0]) + self.armP.l[1]*np.sin(q[0] + q[1])]
        return coordElbow, coordHand
    
    def jacobian(self, q):
        J = np.array([
                    [-self.armP.l[0]*np.sin(q[0]) - self.armP.l[1]*np.sin(q[0] + q[1]),
                     -self.armP.l[1]*np.sin(q[0] + q[1])],
                    [self.armP.l[0]*np.cos(q[0]) + self.armP.l[1]*np.cos(q[0] + q[1]),
                     self.armP.l[1]*np.cos(q[0] + q[1])]])
        return J
    
    def mgdEndEffector(self, q):
        '''
        Direct geometric model of the arm
    
        Inputs:     -q: (2,1) numpy array, the joint coordinates
    
        Outputs:
        -coordHand: hand coordinate
        '''
        coordHand = [self.armP.l[0]*np.cos(q[0])+self.armP.l[1]*np.cos(q[0] + q[1]), 
                     self.armP.l[0]*np.sin(q[0]) + self.armP.l[1]*np.sin(q[0] + q[1])]
        return coordHand


    def estimError(self,state, estimState):
        '''
        Computes a state estimation error as the cartesian distance between the estimated state and the current state
        '''
        q, _ = get_q_and_qdot_from(state)
        qEstim, _ = get_q_and_qdot_from(estimState)
        hand = self.mgdEndEffector(q)
        handEstim = self.mgdEndEffector(qEstim)
        dx = hand[0] - handEstim[0]
        dy = hand[1] - handEstim[1]
        return math.sqrt(dx**2 + dy**2)

    def estimErrorReduced(self,q,qEstim):
        '''
        Computes a state estimation error as the cartesian distance between the estimated state and the current state
        but only taking the position part of the state as input
        '''
        hand = self.mgdEndEffector(q)
        handEstim = self.mgdEndEffector(qEstim)
        dx = hand[0] - handEstim[0]
        dy = hand[1] - handEstim[1]
        return math.sqrt(dx**2 + dy**2)

    def cartesianSpeed(self,state):
        q, qdot = get_q_and_qdot_from(state)
        J = self.jacobian(q)
        return np.linalg.norm(np.dot(J,qdot))


    def mgi(self, xi, yi):
        '''
        Inverse geometric model of the arm
    
        Inputs:     -xi: abscissa of the end-effector point
                    -yi: ordinate of the end-effectior point

        Outputs:
                    -q1: arm angle
                    -q2: foreArm angle
        '''
        a = ((xi**2)+(yi**2)-(self.armP.l[0]**2)-(self.armP.l[1]**2))/(2*self.armP.l[0]*self.armP.l[1])
        try:
            q2 = math.acos(a)
            c = self.armP.l[0] + self.armP.l[1]*(math.cos(q2))
            d = self.armP.l[1]*(math.sin(q2))
            q1 = math.atan2(yi,xi) - math.atan2(d,c)
            return q1, q2
        except ValueError:
            print("forbidden value")
            print (xi,yi)
        return "None"    






    








