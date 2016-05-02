# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:00:39 2016

@author: arnaud
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:00:45 2016

@author: debroissia
"""

from DDPG.core.DDPG_core import DDPG
import numpy as np

import DDPG.environement.instance.mountainCarEnv as mc
from DDPG.core.networks.simple_actor_network import simple_actor_network

import matplotlib.pyplot as plt
from DDPG.logger.result import result_log

l1 = 20
l2 = 10

logger = result_log("DDPG", l1, l2, "simple_"+str(l1)+"_"+str(l2))

env = mc.MountainCarEnv(logger)
a_c = DDPG(env, actor = simple_actor_network(2, 1, l1_size = l1, l2_size = l2, learning_rate = 0.001))
    
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

def perfs():
    env.performances()

def voidFunc():
    pass

env.extern_draw = voidFunc
def draw_buffer():
    for i in range(len(a_c.buffer)):
        plt.scatter((a_c.buffer[i][0][0]+1.)*100, (a_c.buffer[i][0][1]+1.)*100)

def doEp(M, T=float("inf")):
    a_c.M_episodes(M, T)
    env.perfs.save()
