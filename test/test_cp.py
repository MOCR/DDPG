# -*- coding: utf-8 -*-
"""
 A test of DDPG on the pybrain cartpole environment
TODO: replace with the gym cartpole

@author: debroissia
"""

from DDPG.core.DDPG_core import DDPG

import DDPG.environment.instance.cartpoleEnv as cpe


env = cpe.carpoleEnv()
a_c = DDPG(env)

def perfs():
    env.performances()

env.extern_draw = perfs         
    

def doEp(M, T=float("inf")):
    a_c.M_episodes(M, T)
