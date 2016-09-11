# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:00:45 2016

@author: sigaud
"""
import numpy as np

from DDPG.test.helpers.draw import draw_policy, run
from DDPG.core.helpers.Chrono import Chrono

import matplotlib.pyplot as plt

from simple_keras_net import Keras_NN

import gym
env = gym.make('MountainCarContinuous-v0')
#env = gym.make('Pendulum-v0')

layer_nb=2
agent = Keras_NN(2,1)
agent.save_theta("keras_net.save")
glob_scale = 10
        
def draw_varying_policy_1D(agent,numpoids,scale):
    plt.close()
    nb_values = int(200/scale)
    poids = -1.
    save = agent.get_param(layer_nb,numpoids)
    for j in range(nb_values):
        poids += 0.01*scale
        agent.set_param(layer_nb,numpoids,poids)
        global_reward = run(agent,env,False,False)
        plt.scatter(poids,global_reward, c="white")
    agent.set_param(layer_nb,numpoids,save)
    plt.show(block=False)
        
def draw_varying_policy_2D(agent,numpoids1,numpoids2,scale):
    plt.close()
    nb_values = int(200/scale)
    img = np.zeros((nb_values, nb_values))
    poids1 = -1.
    save1 = agent.get_param(layer_nb,numpoids1)
    save2 = agent.get_param(layer_nb,numpoids2)
    for j in range(nb_values):
        poids2 = -1.
        poids1 += 0.01*scale
        for i in range(nb_values):
            poids2 += 0.01*scale
            agent.set_param(layer_nb,numpoids1,poids1)
            agent.set_param(layer_nb,numpoids2,poids2)
            global_reward = run(agent,env,False,False)
            img[i][j] = global_reward

    agent.set_param(layer_nb,numpoids1,save1)
    agent.set_param(layer_nb,numpoids2,save2)
    plt.imshow(img, extent=(-1.0,1.0,-1.0,1.0))
    plt.show(block=False)

def doEp(agent,M):
    agent.perform_M_episodes(M)
    draw_policy(agent,env)

def doInit():
    for i in range(10):
        agent = Keras_NN(2,1)
        draw_policy(agent,env)

def do_weight_study_1D_light(agent):
    c=Chrono()
    nb_poids = agent.get_nb_weights(layer_nb)
    print('nb poids', nb_poids)
    for numpoids in range(nb_poids):
        print('numpoids',numpoids)
        draw_varying_policy_1D(agent,numpoids,glob_scale)
    c.stop()

def do_weight_study_2D(agent):
    nb_poids = agent.get_nb_weights(layer_nb)
    for numpoids1 in range(nb_poids):
        for numpoids2 in range(nb_poids):
            c2=Chrono()
            if numpoids2==numpoids1:
                    numpoids2+=1
            print(numpoids1,"x",numpoids2)
            draw_varying_policy_2D(agent,numpoids1,numpoids2,glob_scale)
            c2.stop()

def load(agent,file_name):
    agent.load_theta(file_name)

def do_full_weight_study(agent):
    nb_poids = agent.get_nb_weights(layer_nb)
    for numpoids in range(nb_poids):
        c=Chrono()
        print('numpoids',numpoids)
        draw_varying_policy_1D(agent,numpoids,glob_scale)
        c.stop()

    for numpoids1 in range(nb_poids):
        for numpoids2 in range(nb_poids):
            c2=Chrono()
            if numpoids2==numpoids1:
                    numpoids2+=1
            print(numpoids1,"x",numpoids2)
            draw_varying_policy_2D(agent,numpoids1,numpoids2,glob_scale)
            c2.stop()

def test(agent):
    draw_policy(agent,env)
    agent.set_param(layer_nb,3,0.1)
    draw_policy(agent,env)

do_weight_study()
