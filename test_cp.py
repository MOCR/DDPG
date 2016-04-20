# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:11:45 2016

@author: arnaud
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:00:45 2016

@author: debroissia
"""

from DDPG import DDPG

import doublecartpoleEnv as cpe


env = cpe.carpoleEnv()
a_c = DDPG(env)

def perfs():
    env.performances()

env.extern_draw = perfs         
    

def doEp(M, T=float("inf")):
    a_c.M_episodes(M, T)
