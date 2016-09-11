import logging
import os, sys

import gym

render=True
max_steps = 1000

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, env):
        self.action_space = env.action_space
        self.env = env
        self.nb_steps = 0
           
    def step(self):
        '''
        perform one step
        nb_steps is the number of steps over this episode
        '''
        action = self.action_space.sample()
        next_state, reward, done, infos = self.env.step(action)
        self.nb_steps += 1
        if render:
            self.env._render()
        self.state = next_state
        return reward, done
    
    def perform_episode(self):
        '''
        perform one episode
        numSteps is the number of steps over all episodes
        '''
        done = False
        self.nb_steps = 0
        while self.nb_steps< max_steps and not done:
            reward, done = self.step()
        return done

    def perform_M_episodes(self, M):
        '''
        perform M episodes
        '''
        done = False
        for i in range(M):
            self.state = self.env.reset()
            done = self.perform_episode()
            print('episode',i,'nbsteps',self.nb_steps)
