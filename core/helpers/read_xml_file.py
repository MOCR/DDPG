#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Olivier Sigaud adapted from Corentin Arnaud

Module: ReadXmlFile

Description: reads a global xml setup file
'''

from lxml import etree

class read_xml_file(object):
    def __init__(self, xmlFile):
        self.parse(xmlFile)

    def parse(self, xmlFile):
        tree = etree.parse(xmlFile).getroot()
        self.parse_actor_params(tree[0])
        self.parse_critic_params(tree[1])
        self.parse_misc_params(tree[2])
    
    def parse_actor_params(self, elem):
        self.actor_l1size=int(elem[0].text)
        self.actor_l2size=int(elem[1].text)
        self.actor_learning_rate=float(elem[2].text)
        self.actor_tracking_rate=float(elem[3].text)
           
    def parse_critic_params(self, elem):
        self.critic_l1size=int(elem[0].text)
        self.critic_l2size=int(elem[1].text)
        self.critic_learning_rate=float(elem[2].text)
        self.critic_tracking_rate=float(elem[3].text)
        self.gamma=float(elem[4].text)
        self.regularization=float(elem[5].text)

    def parse_misc_params(self, elem):
        self.buffer_size=int(elem[0].text)
        self.buffer_min=int(elem[1].text)
        self.minibatch_size=int(elem[2].text)
        self.grad_string=elem[3].text
        self.grad_mode=(self.grad_string=="true")
        self.print_interval=int(elem[4].text)
        self.max_steps=int(elem[5].text)
        self.train_string=elem[6].text
        self.train=(self.train_string=="true")
        self.render_string=elem[7].text
        self.render=(self.render_string=="true")
        self.draw_policy_string=elem[8].text
        self.draw_policy=(self.draw_policy_string=="true")
        self.tensorboard_string=elem[9].text
        self.tensorboard=(self.tensorboard_string=="true")
