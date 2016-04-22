# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:31:15 2016

@author: arnaud
"""

from DDPG.core.DDPG_core import DDPG
import numpy as np

import DDPG.environement.instance.mountainCarEnv as mc


env = mc.MountainCarEnv()
a_c = DDPG(env)

def voidFunc():
    pass

env.extern_draw = voidFunc

def doEp(M, T=float("inf")):
    a_c.M_episodes(M, T)
    env.perfs.save()
    
for i in range(5):
    doEp(2500)