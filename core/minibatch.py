# -*- coding: utf-8 -*-
"""

@author: sigaud
"""
class minibatch(object):
    """
    A minibatch of agent environment interactions
    """
    def __init__(self, s,a,r,s_next):
        self.states=s
        self.actions=a
        self.rewards=r
        self.next_states=s_next
