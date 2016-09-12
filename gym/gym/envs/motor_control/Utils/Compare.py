#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Corentin Arnaud

Module: compare

utils  for comparison
'''

import numpy as np
from scipy.stats import ks_2samp, ranksums, entropy, chisquare, mannwhitneyu
from scipy import minver


def discrete(tab, nb,size):
    tab=np.sort(tab)
    classe= np.zeros(nb) 
    limite= np.linspace(-size,size,nb+1)
    cpt=0
    
    for i in range(tab.shape[0]):
        if(tab[i]>limite[cpt]):
            if(cpt<nb):
                cpt+=1
            else: break
        if(cpt==0): continue
        classe[cpt-1]+=1
    return classe


def chi2(tab1, tab2, size):
    tabD1=discrete(tab1,20,size)
    tabD2=discrete(tab2,20,size)
    #tabD2=tabD2*tabD1.sum()/tabD2.sum()
    return chisquare(tabD1, tabD2)


def ks(tab1, tab2):
    return ks_2samp(tab1, tab2)

def MWWRankSum(tab1, tab2):
    return ranksums(tab1,tab2)

def mw(tab1, tab2):
    return mannwhitneyu(tab1,tab2,"True")

def kl(tab1, tab2, size):
    tabD1=discrete(tab1,100)
    tabD2=discrete(tab2,100)
    #return np.sum(np.where(tabD2 != 0,(tabD1-tabD2) * np.log10(tabD1 / tabD2), 0))
    return entropy(tabD1, tabD2)

