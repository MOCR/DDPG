# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:40:42 2016

@author: debroissia
"""

import random
from collections import deque
import time

from simple_actor_network import simple_actor_network
from simple_critic_network import simple_critic_network
from tensorflow_grad_inverter import grad_inverter
from env import Env



"""A définir"""
def void_noise_func(t):
    return 0.0
def void_env_act(action, store):
    return 0
def void_env_state():
    return [0]
def void_env_ini():
    pass
def void_env_draw():
    pass
def void_env_stop_signial():
    return False

class DDPG(object):
    """First version of DDPG implementation"""
    
    def __init__(self, env):
        self.env = env
        if not isinstance(env,Env):
            print "error"
        
        s_dim = env.getStateSize()
        a_dim = env.getActionSize()
        action_bounds = env.getActionBounds()
        
        self.actor = simple_actor_network(s_dim, a_dim)
        self.critic = simple_critic_network(s_dim, a_dim, action_bounds)

        self.buffer = deque([])
        self.buffer_size = 100000
        self.buffer_minimum = 100
        self.minibatch_size = 64
        self.t = 0
        self.gamma = 0.99
        
        self.grad_inv = grad_inverter(action_bounds)
        
        self.train_loop_size = 1
        self.totStepTime = 0
        self.totTrainTime = 0
        self.batchMix = 0
        self.calcTrain = 0
        self.time0 = 0
        self.time1 = 0
        self.time2 = 0
        self.numStep = 0
        self.stepsTime = 0
    
    def react(self, state):
        act = self.actor.action_batch(state)
        return act
        
    def store_transition(self, s_t, a_t, r_t, s_t_nxt):
        for i in range(len(s_t)):
            if(len(self.buffer)>=self.buffer_size):
                self.buffer[random.randint(self.buffer_size/5, self.buffer_size-1)] = [list(s_t[i]), list(a_t[i]), r_t[i], list(s_t_nxt[i])]
            else:
                self.buffer.append([list(s_t[i]), list(a_t[i]), r_t[i], list(s_t_nxt[i])])
        
            
    def train_Minibatch(self):
        if(len(self.buffer)>self.buffer_minimum):
            rewards = []
            states = []
            nxt_states = []
            actions_batch = []
            batchMix = time.time()
            for i in range(self.minibatch_size):
                r= random.randint(0, len(self.buffer)-1)
                tmp = self.buffer[r]
                rewards.append([tmp[2]])
                states.append(tmp[0])
                nxt_states.append(tmp[3])
                actions_batch.append(tmp[1])
            self.batchMix+=time.time()-batchMix
            
            calcTrain = time.time()
            y = self.critic.y_val_calc(rewards, self.critic.q_val_batch(nxt_states, self.actor.action_batch(nxt_states, True), True))
            self.critic.batch(states, actions_batch, y)
            self.critic.update()
            self.time0 += time.time() - calcTrain
            
            time1 = time.time()
            actions = self.actor.action_batch(states)
            actionGradients = self.grad_inv.invert(self.critic.actionGradient_batch(states, actions), actions)
            self.actor.batch(states, actionGradients)
            self.actor.update()
            self.time1 += time.time() -time1
            
            time2 = time.time()
            self.critic.updateTarget()
            self.actor.updateTarget()
            self.time2+= time.time() -time2
            self.calcTrain+= time.time() - calcTrain
            
    def step(self, train):
        state = list(self.env.state())
        action = self.react(state)
        (action, reward) = self.env.act(action)
        if train:
            self.store_transition(state, action, reward, self.env.state())
        self.t += 1
        return reward
    
    def episode(self, max_t=float("inf"), train=True):
        while self.t<max_t and not self.env.isFinished():
            stepTime = time.time()
            self.step(train)
            if train :
                self.totStepTime += time.time() - stepTime
                for i in range(self.train_loop_size):
                    trainTime = time.time()
                    self.train_Minibatch()
                    self.totTrainTime += time.time() - trainTime
                self.numStep+=1
    def M_episodes(self, M, T=float("inf"), train=True):
        for i in range(M):
            self.t = 0
            if self.env.isFinished():
                self.env.reset()
            self.episode(T, train)
            if i % self.env.print_interval == 0:
                self.stepsTime += self.totStepTime + self.totTrainTime
                self.env.printEpisode()
                self.env.draw()
                #print "step time : ", self.totStepTime/(self.totStepTime+self.totTrainTime) , "train time : ", self.totTrainTime/(self.totStepTime+self.totTrainTime)
                #print "train decomp -> batch mix : ", self.batchMix/(self.batchMix+self.calcTrain), " calculations : ", self.calcTrain/(self.batchMix+self.calcTrain)
                #print "time 0 : ", self.time0/(self.time0 + self.time1+self.time2), "time 1 : ", self.time1/(self.time0 + self.time1+self.time2), "time 2 : ", self.time2/(self.time0 + self.time1+self.time2)                
                #print "Steps/minutes : " , 60.0 / (self.stepsTime / self.numStep)                
                self.totStepTime = 0
                self.totTrainTime = 0
                self.batchMix = 0
                self.calcTrain = 0
                self.time0 = 0
                self.time1 = 0
                self.time2 = 0
#            if i % 1 == 10:
#                self.env.reset(noise=False)
#                self.episode(T, train=False)
#                #self.stepsTime += self.totStepTime + self.totTrainTime
#                self.env.printEpisode()
#                self.env.draw()
#                #print "Steps/minutes : " , 60.0 / (self.stepsTime / self.numStep)                
#                self.totStepTime = 0
#                self.totTrainTime = 0
#                self.batchMix = 0
#                self.calcTrain = 0
#                self.time0 = 0
#                self.time1 = 0
#                self.time2 = 0
    def buffer_flush(self):
        self.buffer = []
            
            
        
    