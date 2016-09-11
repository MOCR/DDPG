# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:00:45 2016

@author: sigaud
"""

from DDPG.test.helpers.draw import draw_policy
from DDPG.test.random_agent import RandomAgent

from DDPG.core.helpers.Chrono import Chrono
from DDPG.logger.result import result_log

import gym
env = gym.make('MountainCarContinuous-v0')
monitor=False

if (monitor):
    env.monitor.start('/home/sigaud/Bureau/sigaud/DDPG_gym/DDPG/log')
    env.monitor.configure(video_callable=lambda count: count % 100 == 0)

#env = gym.make('Pendulum-v0')

l1 = 20
l2 = 10

logger = result_log("DDPG", l1, l2, ""+str(l1)+"_"+str(l2))

def doEp(M):
    agent = RandomAgent(env)
    agent.perform_M_episodes(M)

c=Chrono()
doEp(50)
if (monitor):
    env.monitor.close()
    gym.upload('/home/sigaud/Bureau/sigaud/DDPG_gym/DDPG/log', api_key='sk_oOTW8cLjQIeQeQdalZSApA')
c.stop()
