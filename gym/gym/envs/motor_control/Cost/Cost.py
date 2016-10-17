#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Corentin Arnaud

Module: Cost

Description: Class to compute partial costs
'''
import numpy as np
from gym.envs.motor_control.ArmModel.Arm import get_q_and_qdot_from

class Cost():
    def __init__(self, rs):
        self.rs=rs
        self.previous=0
        self.found=False

    def computeManipulabilityCost(self, arm, state):
        '''
        Computes the manipulability cost on one step of the trajectory
        
        Input:    -cost: cost at time t, float
                
        Output:        -cost: cost at time t+1, float
        '''
        q, qdot = get_q_and_qdot_from(state)
        manip = arm.directionalManipulability(q,self.cartTarget)
        return 1-manip

    def computeStateTransitionCost(self, U):
        '''
        Computes the cost on one step of the trajectory
        
        Input:    -cost: cost at time t, float
                -U: muscular activation vector, numpy array (6,1)
                -t: time, float
                
        Output:        -cost: cost at time t+1, float
        '''
        #compute the square of the norm of the muscular activation vector
        norme = np.linalg.norm(U)
        mvtCost = norme*norme
        #compute the cost following the law of the model
        #return np.exp(-t/self.rs.gammaCF)*(-self.rs.upsCF*mvtCost)
        return -self.rs.upsCF*mvtCost
    
    def computeStateTransitionCostU12(self, U):
        '''
        Computes the cost on one step of the trajectory based on 2 muscles only
        
        Input:    -cost: cost at time t, float
                -U: muscular activation vector, numpy array (6,1)
                -t: time, float
                
        Output:        -cost: cost at time t+1, float
        '''
        #compute the square of the norm of the muscular activation vector
        
        norme = np.linalg.norm(U[:2])
        mvtCost = norme*norme
        #compute the cost following the law of the model
        #return np.exp(-t/self.rs.gammaCF)*(-self.rs.upsCF*mvtCost)
        return -self.rs.upsCF*mvtCost
    
    def computePerpendCost(self, arm, state): 
        '''
        compute the Perpendicular cost for one trajectory
        
        Ouput :        -cost, the perpendicular cost
        ''' 
        q, qdot = get_q_and_qdot_from(state)
        J = arm.jacobian(q)
        xi = np.dot(J,qdot)
        norm=np.linalg.norm(xi)
        if(norm!=0):
            xi = xi/norm
        return 500-1000*xi[0]*xi[0]
    
    def computeHitVelocityCost(self, arm, state): 
        '''
        compute the hit velocity cost for one trajectory
        
        Ouput :        -cost, the hit velocity cost
        ''' 
        speed = arm.cartesianSpeed(state)
        norm = np.linalg.norm(speed)
        return -1000*norm
    
    def compute_reward(self, arm, t, U, i, coordHand, target_size, state):
        done = False
        finished = False
        cost=0
        #cost= (1/(0.01+(coordHand[0]*coordHand[0])+(coordHand[1]-self.rs.YTarget)*(coordHand[1]-self.rs.YTarget)))/100.0
#        if self.previous ==0:
#            cost=0
#        else:
#            cost= 1/((coordHand[0]*coordHand[0])+(coordHand[1]-self.rs.YTarget)*(coordHand[1]-self.rs.YTarget))-self.previous
        #cost = self.computeStateTransitionCost(U)/1000
#        self.previous=1/((coordHand[0]*coordHand[0])+(coordHand[1]-self.rs.YTarget)*(coordHand[1]-self.rs.YTarget))
        
        if coordHand[1] >= self.rs.YTarget:# or coordHand[1] <= 0.2 or coordHand[0]<=-0.3 or coordHand[0]>=0.3:
            finished = True
            self.previous=0
            #check if the Ordinate of the target is reached and give the reward if yes
            if coordHand[1] >= self.rs.YTarget:
                #check if target is reached
                if coordHand[0] >= -target_size/2 and coordHand[0] <= target_size/2:
                    #finished = True
                    cost += 1.0 #np.exp(-t/self.rs.gammaCF)*self.rs.rhoCF/10
                    self.found=True
                    #cost += self.computePerpendCost(arm,state)
#                    cost += self.computeHitVelocityCost(arm,state)
                    print('goal reached')
                    done = True
                else : #if self.found:
                    cost = -0.01#cost+= (1-coordHand[0]*coordHand[0])
            #else:
                #cost += -10000+10000*(coordHand[1]*coordHand[1])
                #q, qdot = get_q_and_qdot_from(state)
        if i>self.rs.max_steps:
            finished=True
            self.previous=0
#                print('xy',coordHand[0],coordHand[1],'-0.6<q1<2.6',q[0],'-0.2<q2<3.0',q[1])
            
        return cost, done, finished
