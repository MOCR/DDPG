# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:40:42 2016

@author: sigaud
"""

import time
import tensorflow as tf

from DDPG.test.helpers.draw import draw_policy

from DDPG.core.sample import sample
from DDPG.core.replay_buffer import replay_buffer
from DDPG.core.noise_generator import noise_generator

from DDPG.core.networks.helper.net_building import create_input_layers
from DDPG.core.networks.helper.fully_connected_network import fully_connected_network
from DDPG.core.networks.helper.mechanics import *
from DDPG.core.networks.helper.operation_sequence import operation_sequence
from DDPG.core.networks.helper.operate import operate

from DDPG.core.networks.helper.network_tracking import track_network, copy_network
from DDPG.core.helpers.read_xml_file import read_xml_file
from DDPG.core.helpers.tensorflow_grad_inverter import grad_inverter

class DDPG_gym(object):
    """
    DDPG's main structure implementation
        env : the task's environment
    """

    def __init__(self, env, config):
        self.env = env
        self.noise_generator = noise_generator()
        self.state = self.env.reset()

        self.config = config

        s_dim = env.observation_space.low.shape[0]
        a_dim = env.action_space.low.shape[0]

        al1s = self.config.actor_l1size
        al2s = self.config.actor_l2size
        cl1s = self.config.critic_l1size
        cl2s = self.config.critic_l2size
        
        fa = [tf.nn.softplus, tf.nn.softplus, None]
        fc = [tf.nn.softplus, tf.nn.tanh, None]
        
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
                                         function=fa,
                                         weight_init_range=weight_init_range_actor)

        self.target_actor = fully_connected_network(next_state_input, 
                                         [al1s, al2s, a_dim], 
                                         function=fa,
                                         trainable=False)
                                         
                                         
        self.critic = fully_connected_network([state_input, action_input],
                                        [cl1s, cl2s, 1], 
                                        function = fc, 
                                        input_layers_connections=input_layers_connections_critic,
                                        weight_init_range=weight_init_range_critic)
        self.critic_a = fully_connected_network([state_input, self.actor.output],
                                        [cl1s, cl2s, 1], 
                                        function = fc, 
                                        input_layers_connections=input_layers_connections_critic,
                                        shared_parameters=self.critic.params)
                                        
        self.target_critic = fully_connected_network([next_state_input, self.target_actor.output],
                                        [cl1s, cl2s, 1], 
                                        function = fc, 
                                        input_layers_connections=input_layers_connections_critic,
                                        trainable=False)
        
        copy_network(self.actor, self.target_actor)
        copy_network(self.critic, self.target_critic)
        
        self.action=operate(self.actor.output, [state_input])
        
        critic_update = minimize_error(self.critic, temporal_difference_error(self.critic, reward_input, self.target_critic.output), self.config.critic_learning_rate)
        
        action_grad = gradient_output_over_tensor(self.critic_a, self.actor.output)
        if self.config.grad_mode:
            action_grad = grad_inverter(action_grad, self.actor.output)
        deterministic_policy_gradient = update_over_output_gradient(self.actor, action_grad, self.config.actor_learning_rate)

        track_actor = track_network(self.actor, self.target_actor, self.config.actor_tracking_rate)
        track_critic = track_network(self.critic, self.target_critic, self.config.critic_tracking_rate)
        
        self.trainer = operation_sequence([critic_update, deterministic_policy_gradient, track_actor + track_critic], inputs)

        self.replay_buffer = replay_buffer(self.config.buffer_min,self.config.buffer_size)

        self.nb_steps = 0
        self.numSteps = 0
        self.stepsTime = 0
    
        self.train_loop_size = 1
        self.totStepTime = 0
        self.totTrainTime = 0

        """
        grad_mod : if you want to do calculations on the action_gradient before
                passing it to the actor (used mainly with gradient inverter to softly bound the actions)
        see: http://arxiv.org/abs/1511.04143 
        """

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
            
    def step(self):
        '''
        perform one step
        nb_steps is the number of steps over this episode
        '''
        action = self.get_noisy_action_from_state(self.state)
        next_state, reward, done, infos = self.env.step(noisy_action)
        self.store_sample(self.state, action, reward, next_state)
        self.render()
        self.state = next_state
        return reward, done

    def render(self):
       if self.config.render:
            self.env._render()
            
    def store_sample(self, state, action, reward, next_state):
        if self.config.train:
            self.replay_buffer.store_one_sample(sample(state, action, reward, next_state))

    def get_noisy_action_from_state(self,state):
        action = self.get_action_from_state(self.state)
        self.nb_steps += 1
        return self.noise_generator.add_noise(action)
    
    def perform_episode(self):
        '''
        perform one episode
        numSteps is the number of steps over all episodes
        '''
        done = False
        while self.nb_steps< self.config.max_steps and not done:
            stepTime = time.time()
            reward, done = self.step()
            self.totStepTime += time.time() - stepTime
            if self.config.train:
                for i in range(self.train_loop_size):
                    trainTime = time.time()
                    self.train()
                    self.totTrainTime += time.time() - trainTime
                self.numSteps+=1
        if self.config.draw_policy:
            draw_policy(self,self.env)
        return done

    def perform_M_episodes(self, M):
        '''
        perform M episodes
        '''
        max_nb_steps = 10000 #OSD:patch to study nb steps
        done = False
        for i in range(M):
            self.nb_steps = 0
            self.state = self.env.reset()
            done = self.perform_episode()
            if i % self.config.print_interval == 0 and self.config.train:
                self.stepsTime += self.totStepTime + self.totTrainTime
                print('nb steps',self.nb_steps)
                print("Steps/minutes : " , 60.0*self.numSteps/self.stepsTime)               
                self.totStepTime = 0
                self.totTrainTime = 0
            if done:
                self.env.reset()
                if (self.nb_steps<max_nb_steps):#OSD:patch to study nb steps
                    max_nb_steps = self.nb_steps
                    print('***** nb steps',self.nb_steps)
                else: print('nb steps',self.nb_steps)

    def train_loop(self):
        if self.config.train:
            for i in range(self.train_loop_size):
                self.train()

    def train(self):
        if (self.replay_buffer.isFullEnough()):
            minibatch = self.replay_buffer.get_random_minibatch(self.config.minibatch_size)
            self.trainer([minibatch.states, minibatch.actions, minibatch.rewards, minibatch.next_states])
