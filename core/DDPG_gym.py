# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:40:42 2016

@author: sigaud
"""

import time
import tensorflow as tf

import pickle

from DDPG.test.helpers.draw import draw_policy

from DDPG.core.sample import sample
from DDPG.core.replay_buffer import replay_buffer
from DDPG.core.noise_generator import noise_generator

from DDPG.core.networks.helper.net_building import create_input_layers
from DDPG.core.networks.helper.fully_connected_network import fully_connected_network
from DDPG.core.networks.helper.mechanics import *
from DDPG.core.networks.helper.operation_sequence import operation_sequence
from DDPG.core.networks.helper.operate import operate

from DDPG.core.networks.helper.network_tracking import track_network
from DDPG.core.helpers.tensorflow_grad_inverter import grad_inverter

from DDPG.core.networks.helper.batch_norm import batch_norm
import numpy as np

def save_DDPG(ddpg_inst, filename):
    f = open(filename, 'w')
    save = {}
    save["actor"]=ddpg_inst.actor.getParams()
    save["critic"]=ddpg_inst.critic.getParams()
    save["config"]=ddpg_inst.config
    pickle.dump(save, f)
    
def load_DDPG(env, filename):
    f = open(filename, 'r')
    save = pickle.load(f)
    return DDPG_gym(env, save["config"], save["actor"], save["critic"])


class DDPG_gym(object):
    """
    DDPG's main structure implementation
        env : the task's environment
    """

    def __init__(self, env, config, actor_parameters = None, critic_parameters = None):
        self.env = env

        self.config = config

        s_dim = env.observation_space.low.shape[0]
        a_dim = env.action_space.low.shape[0]
        
        self.noise_generator = noise_generator(a_dim)

        al1s = self.config.actor_l1size
        al2s = self.config.actor_l2size
        cl1s = self.config.critic_l1size
        cl2s = self.config.critic_l2size
        
        fa = [tf.nn.softplus, tf.nn.softplus, None]
        fc = [tf.nn.softplus, tf.nn.softplus, None]
        
        weight_init_range_actor=[None,None,[-0.0003,0.0003]]
        weight_init_range_critic=[None, None,[-0.0003,0.0003]]
        
        input_layers_connections_critic=[0,1]
    
        inputs = create_input_layers([[None, s_dim],[None, a_dim],[None, 1], [None, s_dim]])
        
        state_input=inputs[0]
        action_input=inputs[1]
        reward_input=inputs[2] 
        next_state_input=inputs[3]

        self.actor = fully_connected_network(state_input, 
                                         [al1s, al2s, a_dim],
                                         normalization=batch_norm,
                                         name="Actor",
                                         function=fa,
                                         weight_init_range=weight_init_range_actor,
                                         cloned_parameters = actor_parameters)

        self.target_actor = fully_connected_network(next_state_input,
                                         [al1s, al2s, a_dim], 
                                         normalization=batch_norm,
                                         name="Target_Actor",
                                         function=fa,
                                         trainable=False,
                                         cloned_parameters = self.actor.params)
                                         
                                         
        self.critic = fully_connected_network([state_input, action_input],
                                        [cl1s, cl2s, 1], 
                                        normalization=batch_norm,
                                        name="Critic",
                                        function = fc, 
                                        input_layers_connections=input_layers_connections_critic,
                                        weight_init_range=weight_init_range_critic,
                                        cloned_parameters = critic_parameters)
                                        
        #this is the same network, with same weight and bias variables, but with a different input layer to allow faster training operations
        self.critic_a = fully_connected_network([state_input, self.actor.output],
                                        [cl1s, cl2s, 1], 
                                        normalization=batch_norm,
                                        name="actor_linked_Critic",
                                        function = fc, 
                                        input_layers_connections=input_layers_connections_critic,
                                        shared_parameters=self.critic.params)
                                        
        self.target_critic = fully_connected_network([next_state_input, self.target_actor.output],
                                        [cl1s, cl2s, 1], 
                                        normalization=batch_norm,
                                        name="Target_Critic",
                                        function = fc, 
                                        input_layers_connections=input_layers_connections_critic,
                                        trainable=False,
                                        cloned_parameters = self.critic.params)
        
        self.action=operate(self.actor.output, [state_input])
        self.value=operate(self.critic.output, [state_input, action_input])
        
        critic_update = minimize_error(self.critic, temporal_difference_error(self.critic, reward_input, self.target_critic.output), self.config.critic_learning_rate)
        
        action_grad = gradient_output_over_tensor(self.critic_a, self.actor.output)
        """
        grad_mod : if you want to do calculations on the action_gradient before
                passing it to the actor (used mainly with gradient inverter to softly bound the actions)
        see: http://arxiv.org/abs/1511.04143 
        """
        if self.config.grad_mode:
            if  hasattr(self.env, 'max_action') and hasattr(self.env, 'min_action'):
                action_bounds=[self.env.max_action, self.env.min_action]
            else:
                action_bounds = None
            action_grad = grad_inverter(action_grad, self.actor.output, action_bounds)
        
        deterministic_policy_gradient = update_over_output_gradient(self.actor, action_grad, self.config.actor_learning_rate)

        track_actor = track_network(self.actor, self.target_actor, self.config.actor_tracking_rate)
        track_critic = track_network(self.critic, self.target_critic, self.config.critic_tracking_rate)
        
        self.trainer = operation_sequence([critic_update, deterministic_policy_gradient,self.actor.updaters + self.critic.updaters, track_actor + track_critic], inputs)

        self.replay_buffer = replay_buffer(self.config.buffer_min,self.config.buffer_size)

        self.nb_steps = 0
        self.numSteps = 0
        self.stepsTime = 0
    
        self.train_loop_size = 1
        self.totStepTime = 0
        self.totTrainTime = 0

    def get_actions_from_batch(self, state_batch):
        """
        Return the actor's action for a batch of states
        """
        return self.action([state_batch])
    
    def get_action_from_state(self, state):
        """
        Return the actor's action for a single states
        TODO: not efficient: uses a batch of one state
        """
        act = self.action([[state]])[0]
        return act

    def get_noisy_action_from_state(self,state):
        action = self.get_action_from_state(state)
        return self.noise_generator.add_noise(action)
            
    def step(self):
        '''
        perform one step
        nb_steps is the number of steps over this episode
        '''
        self.nb_steps += 1
        action = np.array(self.get_noisy_action_from_state(self.state))
        for i in range(len(action)):
            action[i] = max(min(1.0,action[i]) , -1.0)
        next_state, reward, done, infos = self.env.step(action)
        if self.config.train:
            self.store_sample(self.state, action, reward, next_state)
        self.render()
        self.state = next_state
        return reward, done

    def render(self):
       if self.config.render:
            self.env._render()
            
    def store_sample(self, state, action, reward, next_state):
            self.replay_buffer.store_one_sample(sample(state, action, reward, next_state))
    
    def perform_episode(self):
        '''
        perform one episode
        numSteps is the number of steps over all episodes
        '''
        done = False
        totReward = 0
        while self.nb_steps< self.config.max_steps and not done:
            stepTime = time.time()
            reward, done = self.step()
            totReward+=reward
            self.totStepTime += time.time() - stepTime
            if self.config.train:
                for i in range(self.train_loop_size):
                    trainTime = time.time()
                    self.train()
                    self.totTrainTime += time.time() - trainTime
                self.numSteps+=1
        #if (done):
        #    self.noise_generator.decrease_noise()
        #else:
        #    self.noise_generator.increase_noise()            
        if self.config.draw_policy:
            draw_policy(self,self.env)
        return totReward, done

    def perform_M_episodes(self, M):
        '''
        perform M episodes
        '''
        max_nb_steps = 10000 #OSD:patch to study nb steps
        done = False
        for i in range(M):
            self.nb_steps = 0
            self.state = self.env.reset()
            self.noise_generator.randomRange()
            reward, done = self.perform_episode()
            if i%20 == 0:
                self.config.render =True
            else:
                self.config.render =False
            if i % self.config.print_interval == 0 and self.config.train:
                self.stepsTime += self.totStepTime + self.totTrainTime
#                print("Steps/minutes : " , 60.0*self.numSteps/self.stepsTime)               
                self.totStepTime = 0
                self.totTrainTime = 0
            if (self.nb_steps<max_nb_steps):#OSD:patch to study nb steps
                max_nb_steps = self.nb_steps
                print('episode',i,'***** nb steps',self.nb_steps, " perf : ", reward, " total steps :", self.numSteps)
                print(self.replay_buffer.reward_min, self.replay_buffer.reward_max)
            else: print('episode',i,'nb steps',self.nb_steps, " perf : ", reward, " total steps :", self.numSteps)

    def train_loop(self,nb_loops):
        if self.config.train:
            for i in range(nb_loops):
                self.train()

    def train(self):
        if (self.replay_buffer.isFullEnough()):
            minibatch = self.replay_buffer.get_random_minibatch(self.config.minibatch_size)
            self.trainer([minibatch.states, minibatch.actions, minibatch.rewards, minibatch.next_states])
