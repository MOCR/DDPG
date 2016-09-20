# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:00:45 2016

@author: sigaud
"""
import numpy as np

import matplotlib.pyplot as plt

gamma = 0.99

def flatten(l):
    return [item for sublist in l for item in sublist]

def run(agent,env,draw,render):
    state = env.reset()
    steps=0
    global_reward=0
    done = False
    while steps<1000 and not done:
        if (draw):
            plt.scatter(state[0],state[1], c="white")
        action = agent.get_action_from_state(state)
        state, reward, done, infos = env.step(action)
        global_reward+=reward*pow(gamma,steps/1000)
        if (render):
            env._render()
        steps+=1
    return 1000-steps+global_reward

def draw_policy(agent,env):
    plt.close()
    img = np.zeros((200, 200))
    pos = -1.2
    batch = []
    for i in range(200):
        vel = -0.7
        pos += 0.01
        for j in range(200):
            vel += 0.007
            batch.append([pos, vel])
    pol = agent.get_actions_from_batch(batch)
    print ("policy min : ", min(pol)[0], "policy max : ", max(pol)[0])
    b=0           
    for i in range(200):
        for j in range(200):
            img[-j][i] = max(-1, min(1.0, pol[b]))
            b += 1
    plt.imshow(img, extent=(-1.2,0.8,-1.0,1.0))
#    draw_buffer(agent)
    draw_episode(agent,env)
    plt.show(block=True)

def transfo(x,y):
    return x,2*y

def draw_episode(agent,env):
    state = env.reset()
    done = False
    steps=0
    while steps<1000 and not done:
        x,y=transfo(state[0],state[1])
        plt.scatter(x,y, c="white")
        action = agent.get_action_from_state(state)
        state, reward, done, infos = env.step(action)
        steps+=1

def draw_buffer(agent):
    for i in range(agent.replay_buffer.current_size()):
        x,y = transfo(agent.replay_buffer.get_state(i)[0],agent.replay_buffer.get_state(i)[1])
        plt.scatter(x, y, c="green")
