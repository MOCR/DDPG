#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Corentin Arnaud

Module: Cost

Description: Class to compute partial costs
'''
import numpy as np

class Cost():
    def __init__(self, rs):
        self.rs=rs


    def computeManipulabilityCost(self, arm):
        '''
        Computes the manipulability cost on one step of the trajectory
        
        Input:    -cost: cost at time t, float
                
        Output:        -cost: cost at time t+1, float
        '''
        dotq, q = arm.getDotQAndQFromStateVector(self.arm.get_state())
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
    
    def computePerpendCost(self, arm): 
        '''
        compute the Perpendicular cost for one trajectory
        
        Ouput :        -cost, the perpendicular cost
        ''' 
        dotq, q = arm.getDotQAndQFromStateVector(arm.get_state())
        J = arm.jacobian(q)
        xi = np.dot(J,dotq)
        norm=np.linalg.norm(xi)
        if(norm!=0):
            xi = xi/norm
        return 500-1000*xi[0]*xi[0]
    
    def computeHitVelocityCost(self, arm): 
        '''
        compute the hit velocity cost for one trajectory
        
        Ouput :        -cost, the hit velocity cost
        ''' 
        speed = arm.cartesianSpeed(arm.get_state())
        norm = np.linalg.norm(speed)
        return -1000*norm
    
    def compute_reward(self, arm, t, U, i, coordHand, target_size):
        done = False
        finished = False
        cost = self.computeStateTransitionCost(U)

        if coordHand[1] >= self.rs.YTarget or i >= self.rs.max_steps:
            finished = True
            #check if the Ordinate of the target is reached and give the reward if yes
            if coordHand[1] >= self.rs.YTarget:
                #check if target is reached
                if coordHand[0] >= -target_size/2 and coordHand[0] <= target_size/2:
                    cost += np.exp(-t/self.rs.gammaCF)*self.rs.rhoCF
                    cost += self.computePerpendCost(arm)
                    cost += self.computeHitVelocityCost(arm)
                    print('goal reached')
                    done = True
                else:
                    cost+= -50000+50000*(1-coordHand[0]*coordHand[0])
            else:
                cost += -10000+10000*(coordHand[1]*coordHand[1])
            
        return cost, done, finished
