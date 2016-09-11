# -*- coding: utf-8 -*-
"""

@author: sigaud
"""
class sample(object):
    """
    A sample from an agent environment interaction
    """
    def __init__(self, s,a,r,s_next):
        self.state=s
        self.action=a
        self.reward=r
        self.next_state=s_next
