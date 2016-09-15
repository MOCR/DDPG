#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Thomas Beucher

Module: Arm

Description:    
-Models an arm with trhee joints and eight muscles
-Computes its dynamics
'''

import numpy as np
import math

from ArmModel.ArmParamsXML import ArmParameters
from ArmModel.MusclesParamsXML import MusclesParameters
from ArmModel.Arm import Arm

#-----------------------------------------------------------------------------

class Arm38(Arm):
    def __init__(self):
        Arm.__init__(self, ArmParameters(3,8), MusclesParameters(3,8), np.array([0.,0.]))
        self.k1=self.armP.m[2]*(self.armP.l[0]**2)
        self.k2=self.armP.m[1]*self.armP.l[0]*self.armP.s[1]
        self.k3=self.armP.m[1]*(self.armP.l[0]**2)
        self.k4=self.armP.I[0]+self.armP.m[0]*(self.armP.s[0]**2)
        self.k5=self.armP.m[2]*self.armP.l[0]*self.armP.l[1]
        self.k6=self.armP.m[1]*self.armP.s[1]*self.armP.l[0]
        self.k7=self.armP.m[2]*(self.armP.l[1]**2)
        self.k8=self.armP.I[1]+self.armP.m[1]*(self.armP.s[1]**2)
        self.k9=self.armP.m[2]*self.armP.l[0]*self.armP.s[2]
        self.k10=self.armP.m[2]*self.armP.l[1]*self.armP.s[2]
        self.k11=self.armP.I[2]+self.armP.m[2]*(self.armP.s[2]**2)

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
        M = np.array([[self.k4+self.k8+self.k11+self.k3+2*self.k2*math.cos(q[1])+self.k1+self.k7+2*self.k5*math.cos(q[1])+2*self.k9*math.cos(q[1]+q[2])+2*self.k10*math.cos(q[2]),
                       self.k8+self.k11+self.k6*math.cos(q[1])+self.k5*math.cos(q[1])+self.k9*math.cos(q[1]+q[2])+2*self.k10*math.cos(q[2]),
                       self.k11+self.k9*math.cos(q[1]+q[2])+self.k10*math.cos(q[2])],
                  
                      [self.k8+self.k11+self.k6*math.cos(q[1])+self.k5*math.cos(q[1])+self.k9*math.cos(q[1]+q[2])+2*self.k10*math.cos(q[2]),
                       self.k8+self.k11+self.k7+2*self.k10*math.cos(q[2]),
                       self.k11+self.k10*math.cos(q[2])],
                  
                      [self.k11+self.k9*math.cos(q[1]+q[2])+self.k10*math.cos(q[2]),
                       self.k11+self.k10*math.cos(q[2]),
                       self.k11]])
        

        C = np.array([-self.k2*qdot[0]*qdot[1]*math.sin(q[1])-self.k5*math.sin(q[1])-self.k9*math.sin(q[1]+q[2])*(2*qdot[0]+qdot[1]+qdot[2])
                      -self.k10*math.sin(q[2])*(2*qdot[0]*qdot[2]+2*qdot[1]*qdot[2]+qdot[2]**2)-self.k6*math.sin(q[2])*(qdot[3]**2),
                                           
                      -(qdot[0]**2)*math.sin(q[1])*(self.k5+self.k2)+self.k9*(qdot[0]**2)*math.sin(q[1]+q[2])
                      +self.k10*qdot[2]*math.sin(q[2])*(-2*qdot[0]-2*qdot[1]-qdot[2]),
                      
                      self.k9*(qdot[0]**2)*math.sin(q[1]+q[2])+self.k10*math.sin(q[2])*((qdot[0]+qdot[2])**2)])
        #print ("C:",C)
        #the commented version uses a non null stiffness for the muscles
        #beware of dot product Kraid times q: q may not be the correct vector/matrix
        #Gamma = np.dot((np.dot(armP.At, musclesP.fmax)-np.dot(musclesP.Kraid, q)), U)
        #Gamma = np.dot((np.dot(self.armP.At, self.musclesP.fmax)-np.dot(self.musclesP.Knulle, Q)), U)
        #above Knulle is null, so it can be simplified

        Gamma = np.dot(np.dot(self.armP.At, self.musclesP.fmax), U)
        #print ("Gamma:",Gamma)

        #computes the acceleration qddot and integrates
    
        b = np.dot(self.armP.B, qdot)
        #print ("b:",b)

        #To avoid inverting M:
        qddot = np.linalg.solve(M, Gamma - C - b)


        #print ("qddot",qddot)

        qdot += qddot*self.dt
        #print ("qdot",qdot)
        q += qdot*self.dt
        #save the real state to compute the state at the next step with the real previous state
        q = self.joint_stop(q)
        nextState = np.array([qdot[0], qdot[1], qdot[2], q[0], q[1], q[2]])
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
        coordWrist = [self.armP.l[0]*np.cos(q[0])+self.armP.l[1]*np.cos(q[0] + q[1]), self.armP.l[0]*np.sin(q[0]) + self.armP.l[1]*np.sin(q[0] + q[1])]
        coordHand = [self.armP.l[0]*np.cos(q[0])+self.armP.l[1]*np.cos(q[0]+q[1])+self.armP.l[2]*np.cos(q[0]+q[1]+q[2]), self.armP.l[0]*np.sin(q[0])+self.armP.l[1]*np.sin(q[0]+q[1])+self.armP.l[2]*np.sin(q[0]+q[1]+q[2])]
        return coordElbow, coordWrist, coordHand
    
    def jacobian(self, q):
        J = np.array([[-self.armP.l[0]*np.sin(q[0])-self.armP.l[1]*np.sin(q[0]+q[1])-self.armP.l[2]*np.sin(q[0]+q[1]+q[2]), -self.armP.l[1]*np.sin(q[0]+q[1])-self.armP.l[2]*np.sin(q[0]+q[1]+q[2]), -self.armP.l[2]*np.sin(q[0]+q[1]+q[2])], 
                    [self.armP.l[0]*np.cos(q[0])+self.armP.l[1]*np.cos(q[0]+q[1])+self.armP.l[2]*np.cos(q[0]+q[1]+q[2]), self.armP.l[1]*np.cos(q[0]+q[1])+self.armP.l[2]*np.cos(q[0]+q[1]+q[2]), self.armP.l[2]*np.cos(q[0]+q[1]+q[2])]])
        return J

    def mgdEndEffector(self, q):
        '''
        Direct geometric model of the arm
    
        Inputs:     -q: (3,1) numpy array, the joint coordinates
    
        Outputs:
                    -coordHand: hand coordinate
        '''
        #print q
        coordHand = [self.armP.l[0]*np.cos(q[0])+self.armP.l[1]*np.cos(q[0]+q[1])+self.armP.l[2]*np.cos(q[0]+q[1]+q[2]), self.armP.l[0]*np.sin(q[0])+self.armP.l[1]*np.sin(q[0]+q[1])+self.armP.l[2]*np.sin(q[0]+q[1]+q[2])]
        return coordHand

    def mgi(self, xi, yi):
        '''
        Inverse geometric model of the arm
    
        Inputs:     -xi: abscissa of the end-effector point
                    -yi: ordinate of the end-effectior point

        Outputs:
                    -q1: arm angle
                    -q2: foreArm angle
                    -q3: hand angle 
        '''
        a = (xi**2+yi**2-self.armP.l[0]**2-(self.armP.l[1]+self.armP.l[2])**2)/(2*self.armP.l[0]*(self.armP.l[1]+self.armP.l[2]))
        try:
            q3 = 0
            q2 = math.acos(a)
            b = (yi*(self.armP.l[0]+(self.armP.l[1]+self.armP.l[2])*math.cos(q2))-xi*(self.armP.l[1]+self.armP.l[2])*math.sin(q2))/(xi*(self.armP.l[0]+(self.armP.l[1]+self.armP.l[2])*math.cos(q2))+yi*(self.armP.l[1]+self.armP.l[2])*math.sin(q2))
            q1 = math.atan(b)
            return q1, q2, q3
        except ValueError:
            print("forbidden value")
            print xi,yi
        return "None"    
