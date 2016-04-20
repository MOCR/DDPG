# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:40:58 2016

@author: arnaud
"""

from mountainCarEnv import MountainCarEnv
from simple_actor_network import simple_actor_network
from result import result_log
import matplotlib.pyplot as plt
import numpy as np

import cma
import DDPG


def CMA_obj(x, *args):
    CMA_obj.act.load_parameters(x)
    CMA_obj.env.reset(noise = False)
    ret = 0.0
    while not CMA_obj.env.isFinished():
        actn, r = CMA_obj.env.act(CMA_obj.act.action_batch(CMA_obj.env.state()))
        ret += r[0]
        CMA_obj.totStep += 1
    return 100-ret
    
def term_callback(c):
    if CMA_obj.totStep != term_callback.lastCall:
        term_callback.lastCall = CMA_obj.totStep
        CMA_obj.act.load_parameters(c.mean)
        CMA_obj.env.reset(noise = False)
        ret = 0
        while not CMA_obj.env.isFinished():
            actn, r = CMA_obj.env.act(CMA_obj.act.action_batch(CMA_obj.env.state()))
            ret += r[0]
        term_callback.plot_data.addData(CMA_obj.totStep, CMA_obj.env.t, ret)
    return False

CMA_obj.env = MountainCarEnv()
CMA_obj.act = simple_actor_network(2, 1)
CMA_obj.totStep = 0

term_callback.plot_data = result_log("CMA-ES", 20,10)
term_callback.lastCall = -1
print "Going for CMA-ES"
op = cma.CMAOptions()
op['termination_callback'] = term_callback
op['maxiter']=200
x=  CMA_obj.act.linear_parameters()
fx = cma.fmin(CMA_obj, x, 0.5, options = op)

term_callback.plot_data.save()
#print fx[0]
a_c = DDPG.DDPG(CMA_obj.env)
a_c.actor.load_parameters(fx[0])

def draw_politic():
    plt.close()
    ac= a_c
    img = np.zeros((200, 200))
    pos = -1.
    batch = []
    for i in range(200):
        vel = -1.
        pos += 0.01
        for j in range(200):
            vel += 0.01
            batch.append([pos, vel])
    pol = ac.react(batch)
    b=0           
    print "politic max : ", max(pol), " politic min : ", min(pol)
    for i in range(200):
        for j in range(200):
            img[j][i] = max(-1, min(1.0, pol[b]))
            b += 1
    img[0][0] = -1
    img[-1][-1] = 1.0
    plt.imshow(img)
    plt.show(block=False)
    
#draw_politic()
CMA_obj.env.extern_draw = draw_politic

def doEp(M, T=float("inf")):
    a_c.M_episodes(M, T, train=False)
