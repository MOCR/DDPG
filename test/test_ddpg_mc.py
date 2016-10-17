# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:00:45 2016

@author: sigaud
"""

#from DDPG.test.helpers.draw import draw_policy

from DDPG.core.DDPG_gym import DDPG_gym, save_DDPG
from DDPG.core.helpers.Chrono import Chrono
#from DDPG.logger.result import result_log

from DDPG.core.helpers.read_xml_file import read_xml_file

import gym
config = read_xml_file("DDPG_mc_config.xml")

#env = gym.make('MountainCarContinuous-v0')
env = gym.make('LunarLanderContinuous-v2')
#env.configure(deterministic=False)

#env = gym.make('Pendulum-v0')
#env = gym.make('Acrobot-v0')

monitor=False

if (monitor):
    env.monitor.start('/home/sigaud/Bureau/sigaud/DDPG_gym/DDPG/log',force=True)
    env.monitor.configure(video_callable=lambda count: count % 100 == 0)

agent = DDPG_gym(env,config)

def doEp(M):
    agent.perform_M_episodes(M)
    #draw_policy(agent,env)

def doInit():
    for i in range(10):
        agent = DDPG_gym(env)
 #       draw_policy(agent,env)

c=Chrono()
doEp(10000)
save_DDPG(agent, "Agent_0.ddpg")

if (monitor):
    env.monitor.close()
    gym.upload('/home/sigaud/Bureau/sigaud/DDPG_gym/DDPG/log', api_key='sk_oOTW8cLjQIeQeQdalZSApA')
c.stop()
