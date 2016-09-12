# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:40:58 2016

@author: sigaud
"""
import numpy as np

import cma
import sys
import math

from DDPG.test.helpers.draw import draw_policy, run
from DDPG.core.helpers.Chrono import Chrono

from simple_keras_net import Keras_NN

import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from matplotlib import cm

plt.rc("figure", facecolor="white")

import gym

#env = gym.make('MountainCarContinuous-v0')
env = gym.make('Pendulum-v0')
#env = gym.make('Acrobot-v0')
layer_nb = 4
sigma = 0.05

numpoids1 = 2
numpoids2 = 4
scale = 1

class CMA():

    def __init__(self,env):
        '''
	Input:
        '''
        self.env = env
        self.env.reset(False)
        self.controller = Keras_NN(env.observation_space.low.shape[0],env.action_space.low.shape[0])
        self.controller.load_theta('cma_agents/best_agent.theta-2772.2691621')
        self.nb_episodes = 0
        self.options = cma.CMAOptions()
        self.options['maxiter']=40
        self.options['popsize']=50
#        self.options['CMA_diagonal']=True
        self.options['verb_log']=50
        self.options['verb_disp']=1
        self.best_perf = -10000.0

        self.start_x = 0.73
        self.start_y = -0.26
        self.controller.set_param(layer_nb, numpoids1, self.start_x)
        self.controller.set_param(layer_nb, numpoids2, self.start_y)
        
        self.white_x = []
        self.white_y = []
        
        self.best_x = []
        self.best_y = []
        
        self.green_x = []
        self.green_y = []

        self.xmin = 10000
        self.ymin = 10000
        self.xmax = -10000
        self.ymax = -10000

    def final_draw(self):
        plt.close()
        self.fig = plt.figure(1, figsize=(16,9))

        self.draw_cost(self.controller,numpoids1,numpoids2,scale)
        
        plt.scatter(self.best_x,self.best_y, c="white")
        plt.plot(self.best_x,self.best_y, '-x',c="white")

        plt.scatter(self.green_x,self.green_y, c="green")
        
#        plt.scatter(self.white_x,self.white_y, c="white")

        plt.scatter(self.start_x,self.start_y, c="white")
        plt.scatter(self.final_x,self.final_y,c="cyan")
        
        print(self.xmin,self.xmax,self.ymin,self.ymax)

#        fig.colorbar(plt, shrink=0.5, aspect=5)
        
        plt.xlabel("w1")
        plt.ylabel("w2")
        plt.title("Cost map")

        plt.savefig('study.svg', bbox_inches='tight')
        plt.show(block=True)

    def draw_cost(self,agent,numpoids1,numpoids2,scale):

        self.xmin = -1.5
        self.ymin = -0.5
        self.xmax = 1.5
        self.ymax = 1.5

        w1 = self.xmin
        save1 = agent.get_param(layer_nb,numpoids1)
        save2 = agent.get_param(layer_nb,numpoids2)
        x_range = self.xmax -self.xmin
        y_range = self.ymax -self.ymin
        scale = max(1, scale * max(x_range,y_range)/2)
        nb_valuesx = int(x_range*100/scale)
        nb_valuesy = int(y_range*100/scale)
           
        x0 = []
        y0 = []
        cost = []
        
        for i in range(nb_valuesx):
            w2 = self.ymin
            w1 += 0.01*scale
            for j in range(nb_valuesy):
                w2 += 0.01*scale
                agent.set_param(layer_nb,numpoids1,w1)
                agent.set_param(layer_nb,numpoids2,w2)
                global_reward = run(agent,self.env,False,False)
                x0.append(w1)
                y0.append(w2)
                cost.append(global_reward)

        agent.set_param(layer_nb,numpoids1,save1)
        agent.set_param(layer_nb,numpoids2,save2)

        xi = np.linspace(self.xmin,self.xmax,100)
        yi = np.linspace(self.ymin,self.ymax,100)
        zi = griddata(x0, y0, cost, xi, yi)
    
        t1 = plt.scatter(x0, y0, c=cost, marker=u'o', s=5)#cmap=cm.get_cmap('RdYlBu'))
        plt.contourf(xi, yi, zi, 15)#, cmap=cm.get_cmap('RdYlBu'))
        self.fig.colorbar(t1, shrink=0.5, aspect=5)
        #plot the interpolation points
        #t1 = plt.scatter(x0, y0, c='b', marker=u'o', s=5)

    def draw_cost_landscape(self,agent,numpoids1,numpoids2,scale):
        x_range = self.xmax -self.xmin
        y_range = self.ymax -self.ymin
        scale = max(1, scale * max(x_range,y_range)/2)
        nb_valuesx = int(x_range*100/scale)
        nb_valuesy = int(y_range*100/scale)
        
        img = np.zeros((nb_valuesx, nb_valuesy))
        poids1 = self.xmin
        save1 = agent.get_param(layer_nb,numpoids1)
        save2 = agent.get_param(layer_nb,numpoids2)
        
        for i in range(nb_valuesx):
            poids2 = self.ymin
            poids1 += 0.01*scale
            for j in range(nb_valuesy):
                poids2 += 0.01*scale
                agent.set_param(layer_nb,numpoids1,poids1)
                agent.set_param(layer_nb,numpoids2,poids2)
                global_reward = run(agent,self.env,False,False)
                img[i][j] = global_reward

        agent.set_param(layer_nb,numpoids1,save1)
        agent.set_param(layer_nb,numpoids2,save2)
        plt.imshow(img, extent=(self.xmin,self.xmax,self.ymin,self.ymax))

    def run_episode(self,x):
        w1 = x[0]
        w2 = x[1]
        self.controller.set_param(layer_nb, numpoids1, w1)
        self.controller.set_param(layer_nb, numpoids2, w2)
        if w1>self.xmax: self.xmax = w1
        if w1<self.xmin: self.xmin = w1
        if w2>self.ymax: self.ymax = w2
        if w2<self.ymin: self.ymin = w2
        self.nb_episodes += 1
        perf = run(self.controller, self.env, False, False)
        if (perf>self.best_perf):
            self.best_x.append(w1)
            self.best_y.append(w2)
            self.best_perf=perf
            print('w1',w1,'w2',w2)
            self.final_x = w1
            self.final_y = w2
            self.controller.save_theta('cma_agents/best_cma.theta'+str(perf))
            print('episode',self.nb_episodes, 'perf', perf)
        else:
            self.white_x.append(w1)
            self.white_y.append(w2)
        return -perf

    def call_cma(self):
        w1=  self.controller.get_weight(layer_nb, numpoids1)[0]
        w2=  self.controller.get_weight(layer_nb, numpoids2)[0]
        x = np.array([w1,w2])
        self.green_x.append(w1)
        self.green_y.append(w2)
        fx = cma.fmin(self.run_episode, x, sigma, options = self.options)
        self.final_draw()
        
optim = CMA(env)
optim.call_cma()
