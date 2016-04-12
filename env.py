# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:41:20 2016

@author: arnaud
"""

""" 
This is the class that any environement should inherit from,
it is the main piece to interact with the DDPG algorithme.
"""

def void_env_draw():
    pass


class Env:
    """
    extern_draw is a function that is used when draw is called, change it to a custom
    function
    """
    extern_draw = void_env_draw
    """
    print_interval specify how many episode is to be done between each call of printEpisode
    set to float("inf") to never print
    """
    print_interval = 1
    """
    noise_func is used to add noise to action.
    Return is a list of noises (float) the size of the actions
    """
    def noise_func(self):
        return [0.0]
    """
    getActionSize
    return the number of action dimensions.
    """
    def getActionSize(self):
        return 1
    """
    getStateSize
    return the number of state dimensions
    """
    def getStateSize(self):
        return 1
    """
    getActionBounds :
    return the upper and lower bounds of all the action dimensions (it is recommended to make them a bit larger than the actual ones)
    return is a list containing a first list for the upper bounds and a second list for the lower bounds.
    """
    def getActionBounds(self):
        return [[1], [-1]]
    """
    act : used to perform an action and get the corresponding reward.
    action have the shape list(list(float)), the outter list containing one element.
    action noise should be added here.
    return the noised actions and the associated reward
        noised actions should be a list containing one list of all the action dimentions
        reward is a list of one float that represent the reward
    """
    def act(self, action):
        return [[0]], [0]
    """
        state : used to get the current state of the environement
            return a list containing one list that contains floats representing all the state dimensions
    """
    def state(self):
        return [[0]]
    """
        reset : used to reset the environement and start a new episode.
            optional parameter : noise (bool) specify if actions should be noised durring the episode
    """
    def reset(self, noise=True):
        print "wrong reset"
        pass
    """
        draw : used to draw some results, called at the same time as printEpisode
    """
    def draw(self):
        self.extern_draw()
    """
        isFinished : return a bool representing if the episode is finished and the environement should be reset
    """
    def isFinished(self):
        return False
    """
        printEpisode : used to print results at each 'print_interval' number of episodes done.
    """
    def printEpisode(self):
        pass