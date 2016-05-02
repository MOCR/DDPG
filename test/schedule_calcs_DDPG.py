# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:31:15 2016

@author: arnaud
"""

from DDPG.core.DDPG_core import DDPG
import numpy as np

import DDPG.environement.instance.mountainCarEnv as mc
from DDPG.core.networks.simple_actor_network import simple_actor_network
from DDPG.logger.result import result_log

l1 = 20
l2 = 10
rate = 0.001

env = mc.MountainCarEnv(result_log("DDPG", l1, l2))
a_c = DDPG(env, actor = simple_actor_network(2, 1, l1_size = l1, l2_size = l2, learning_rate = rate))

def voidFunc():
    pass

env.extern_draw = voidFunc

def doEp(M, T=float("inf")):
    a_c.M_episodes(M, T)
    env.perfs.save()
doEp(4000)