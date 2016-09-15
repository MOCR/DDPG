# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:00:45 2016

@author: sigaud
"""

from DDPG.test.helpers.draw import draw_policy

from DDPG.core.DDPG_gym import DDPG_gym, load_DDPG
from DDPG.core.helpers.Chrono import Chrono
from DDPG.logger.result import result_log

import gym

env = gym.make('MountainCarContinuous-v0')
env.configure(deterministic=False)

#env = gym.make('Pendulum-v0')
#env = gym.make('Acrobot-v0')

l1 = 20
l2 = 10

logger = result_log("DDPG", l1, l2, ""+str(l1)+"_"+str(l2))
agent = load_DDPG(env,"Agent_0.ddpg")

def doEp(M):
    agent.perform_M_episodes(M)
    draw_policy(agent,env)

def doInit():
    for i in range(10):
        agent = DDPG_gym(env)
        draw_policy(agent,env)

c=Chrono()
doEp(100)
