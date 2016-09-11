# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:00:45 2016

@author: sigaud
"""
import numpy as np
import pybrain

from DDPG.test.helpers.draw import draw_policy, run
from DDPG.core.helpers.Chrono import Chrono

import matplotlib.pyplot as plt

from pybrain_net import NeuralNetPB

import gym
env = gym.make('MountainCarContinuous-v0')
#env = gym.make('Pendulum-v0')
        
def draw_varying_policy(agent,numpoids):
    plt.close()
    poids = -1.
    save = agent.get_param(numpoids)
    for j in range(200):
        poids += 0.01
        agent.set_param(numpoids,poids)
        global_reward = run(agent,env,False,False)
        plt.scatter(poids,global_reward, c="white")
    agent.set_param(numpoids,save)
    plt.show(block=False)
        
def draw_varying_policy_light(agent,numpoids):
    plt.close()
    poids = -1.
    save = agent.get_param(numpoids)
    for j in range(100):
        poids += 0.02
        agent.set_param(numpoids,poids)
        global_reward = run(agent,env,False,False)
        plt.scatter(poids,global_reward, c="white")
    agent.set_param(numpoids,save)
    plt.show(block=False)
        
def draw_varying_policy2(agent,numpoids1,numpoids2):
    plt.close()
    img = np.zeros((200, 200))
    poids1 = -1.
    save1 = agent.get_param(numpoids1)
    save2 = agent.get_param(numpoids2)
    for j in range(200):
        poids2 = -1.
        poids1 += 0.01
        for i in range(200):
            poids1 += 0.01
            agent.set_param(numpoids1,poids1)
            agent.set_param(numpoids2,poids2)
            global_reward = run(agent,env,False,False)
            img[-j][i] = global_reward

    agent.set_param(numpoids1,save1)
    agent.set_param(numpoids2,save2)
    plt.imshow(img, extent=(-1.0,1.0,-1.0,1.0))
    plt.show(block=False)
        
def draw_varying_policy2_light(agent,numpoids1,numpoids2):
    plt.close()
    img = np.zeros((50, 50))
    poids1 = -1.
    save1 = agent.get_param(numpoids1)
    save2 = agent.get_param(numpoids2)
    for j in range(50):
        poids2 = -1.
        poids1 += 0.04
        for i in range(50):
            poids1 += 0.04
            agent.set_param(numpoids1,poids1)
            agent.set_param(numpoids2,poids2)
            global_reward = run(agent,env,False,False)
            img[-j][i] = global_reward

    agent.set_param(numpoids1,save1)
    agent.set_param(numpoids2,save2)
    plt.imshow(img, extent=(-1.0,1.0,-1.0,1.0))
    plt.show(block=False)

def doEp(M):
    agent = NeuralNetPB(2,1)
    agent.perform_M_episodes(M)
    draw_policy(agent,env)

def doInit():
    for i in range(10):
        agent = NeuralNetPB(2,1)
        draw_policy(agent,env)

def do_perf_weight():
    agent = NeuralNetPB(2,1)
    for numpoids in range(len(agent.net.params)):
        draw_varying_policy(agent,numpoids)

def do_perf_weight2():
    agent = NeuralNetPB(2,1)
    for numpoids1 in range(len(agent.net.params)):
        for numpoids2 in range(len(agent.net.params)):
            draw_varying_policy2(agent,numpoids1,numpoids2)


def do_weight_study():
    agent = NeuralNetPB(2,1)
    print agent.net.params
    agent.save_theta("pybrain_net.save")
    agent.load_theta("pybrain_net.save")
    print agent.net.params
    for numpoids in range(len(agent.net.params)):
        c=Chrono()
        print(numpoids)
        draw_varying_policy_light(agent,numpoids)
        c.stop()
    for numpoids1 in range(len(agent.net.params)):
        for numpoids2 in range(len(agent.net.params)):
            c2=Chrono()
            if numpoids2==numpoids1:
                    numpoids2+=1
            print(numpoids1,"x",numpoids2)
            draw_varying_policy2_light(agent,numpoids1,numpoids2)
            c2.stop()
