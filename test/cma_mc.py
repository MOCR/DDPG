# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:40:58 2016

@author: arnaud
"""

from DDPG.environment.instance.mountainCarEnv import MountainCarEnv
from DDPG.core.networks.simple_actor_network import simple_actor_network
from DDPG.logger.result import result_log

import cma
import sys
#import DDPG.core.DDPG_core
import math

l1 = int(sys.argv[1])
l2 = int(sys.argv[2])


def CMA_obj(x, *args):
    CMA_obj.act.load_parameters(x)
    CMA_obj.env.reset(noise = False)
    ret = 0.0
    step = 0
    while not CMA_obj.env.isFinished():
        
        actn, r = CMA_obj.env.act(CMA_obj.act.action_batch(CMA_obj.env.state()))
        ret += r[0]*math.pow(0.99, step)
        step += 1
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
CMA_obj.act = simple_actor_network(2, 1, l1_size = l1, l2_size = l2)
CMA_obj.totStep = 0

term_callback.plot_data = result_log("CMA-ES", 20,10)
term_callback.lastCall = -1
print "Going for CMA-ES"
op = cma.CMAOptions()
#op['termination_callback'] = term_callback
op['maxiter']=200
x=  CMA_obj.act.linear_parameters()
fx = cma.fmin(CMA_obj, x, 0.5, options = op)

term_callback.plot_data.save()
#    
##draw_politic()
#CMA_obj.env.extern_draw = draw_politic
#
#def doEp(M, T=float("inf")):
#    a_c.M_episodes(M, T, train=False)
