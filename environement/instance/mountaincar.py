__author__ = 'Tom Schaul, tom@idsia.ch'

"""
Adaptation of the MountainCar Environment
from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0).
"""
    
from scipy import cos
from pybrain.rl.environments.episodic import EpisodicTask

import random
import math


class MountainCar(EpisodicTask): 
    # The current real values of the state
    cur_pos = -0.5
    cur_vel = 0.0
    cur_state = [cur_pos, cur_vel]
    
    nsenses = 3

    # number of steps of the current trial
    steps = 0

    # number of the current episode
    episode = 0

    # Goal Position
    goalPos = 0.45
    
    maxSteps = 999
    
    resetOnSuccess = True

    def __init__(self):
        self.reset()
        self.cumreward = 0
        self.maxpos = 0
        self.minpos = 0

    def reset(self):
        self.state = self.GetInitialState()
    
    def getObservation(self):    
        #print(array([self.state[0], self.state[1] * 100, 1]))
        return [(self.state[0]+0.5), (self.state[1]/7.0)*100]
        
    def performAction(self, action):
        if self.done > 0:
            self.done += 1            
        else:
            self.state = self.DoAction(action, self.state)
            self.r, self.done = self.GetReward(self.state, action)
            self.cumreward += self.r
            
    def getReward(self):
        return self.r    

    def GetInitialState(self):
        self.StartEpisode()
        p = -0.5 + random.uniform(-0.6,0.6)*0
        v = 0.0 + random.uniform(-0.06,0.06)*0
        return [p,v]

    def StartEpisode(self):
        self.steps = 0
        self.episode = self.episode + 1
        self.done = 0
        
    def isFinished(self):
        if self.done>=1 and self.resetOnSuccess:
            self.reset()
            return True
        else:
            return self.done>=3
    

    def GetReward(self, s, a):
        # MountainCarGetReward returns the reward at the current state
        # x: a vector of position and velocity of the car
        # r: the returned reward.
        # f: true if the car reached the goal, otherwise f is false

        position = s[0]
        vel = s[1]
        # bound for position; the goal is to reach position = 0.45
        bpright = self.goalPos

        r = 0
        f = 0
        
        if  position >= bpright:
            r = 100#-self.steps/10.0
            f = 1
#        elif position>self.maxpos:
#            r = 5
#            self.maxpos = position
#        elif position<self.minpos:
#            r = 5
#            self.minpos = position
        #r += math.sqrt(math.pow(position+0.5, 2) +math.pow(vel*20, 2))/10.0
            
        if self.steps >= self.maxSteps:
            f = 5
            
        r-= math.pow(a,2)*0.1
        #r+= math.pow(vel*100/7.0, 2)
#        if a>1.1 or a<-1.1:
#            r += -10.0

        return r, f

    def DoAction(self, a, s):
        #MountainCarDoAction: executes the action (a) into the mountain car
        # acti: is the force to be applied to the car
        # x: is the vector containning the position and speed of the car
        # xp: is the vector containing the new position and velocity of the car
        #print('action',a)
        #print('state',s)
#        if a[0]>0.1:
#            force = 1
#        elif a[0]<-0.1:
#            force = -1
#        else:
#            force = 0
        force = min(max(a[0], -1.0), 1.0)

        self.steps = self.steps + 1

        position = s[0]
        speed = s[1]
        #print position, speed

        # bounds for position
        bpleft = -1.4

        # bounds for speed
        bsleft = -0.07
        bsright = 0.07

        speedt1 = speed + (0.0015 * force) + (-0.0025 * cos(3.0 * position))
        #print speedt1
        
        if speedt1 < bsleft:
            speedt1 = bsleft
        elif speedt1 > bsright:
            speedt1 = bsright

        post1 = position + speedt1

        if post1 <= bpleft:
            post1 = bpleft
            speedt1 = 0.0
        #print post1, speedt1
        return [post1, speedt1]