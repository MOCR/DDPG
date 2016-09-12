# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:40:58 2016

@author: sigaud
"""

import gym

from DDPG.test.cma_agent import CMA
env = gym.make('Pendulum-v0')
#env = gym.make('MountainCarContinuous-v0')
#env = gym.make('Acrobot-v0')

optim = CMA(env)
optim.call_cma()
