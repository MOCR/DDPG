#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 15 févr. 2016

@author: Corentin Arnaud 
'''
from shutil import copyfile
import os
import numpy as np

def check_if_theta_file_exists(foldername,target_size,num):
    dir = foldername + str(target_size) + "/" + str(num) + "/Theta/"
    return os.path.isdir(dir)

def count_best_files(foldername):
    retour = []
    for target_size in [0.005, 0.01, 0.02, 0.04]:
        count=0
        for i in range(15):
            if check_if_theta_file_exists(foldername,target_size,i):
                count+=1
        retour.append(count)
    return retour

def show_count_best_files(foldername):
    count = count_best_files(foldername)
    i=0
    for target_size in [0.005, 0.01, 0.02, 0.04]:
        print("target =",target_size,"nb_theta:",count[i])
        i += 1

def count_best_files_over_gamma(foldername):
    for gamma in range(3,10):
        fullfoldername = foldername + str(gamma) + "/"
        count=count_best_files(fullfoldername)
        total=0
        for i in range(4):
            total += count[i]
        print("gamma =",gamma,"nbtheta:",count,"total:",total)

def find_best_theta_file(fullfoldername):
    if not os.path.isdir(fullfoldername): return 0.0
    best_perf=0.0
    for name in os.listdir(fullfoldername):
        perf = name[11:]
#        print ('foldername',fullfoldername,'name',name,'perf',perf)
        perf = float(perf)
        if perf>best_perf:
            best_perf=perf
    return best_perf

def show_perfs(foldername):
    for target_size in [0.005, 0.01, 0.02, 0.04]:
        for num in range(15):
            if check_if_theta_file_exists(foldername,target_size,num):
                fullfoldername = foldername + str(target_size) + "/" + str(num) + "/Theta/"
                perf = find_best_theta_file(fullfoldername)
                print(target_size, num, perf)
            else:
                print(target_size, num,': **********')#,foldername)
        print('--------------------------------')

def checkIfFolderExists(name):
    if not os.path.isdir(name):
        os.makedirs(name)

def findDataFilename(foldername, name, extension):
    i = 1
    checkIfFolderExists(foldername)
    tryName = name + "1" + extension
    while tryName in os.listdir(foldername):
        i += 1
        tryName = name + str(i) + extension
    filename = foldername + tryName
    return filename

def copyRegressiontoCMAES(rs, name, size):
    cmaname =  rs.CMAESpath + str(size) + "/"
    checkIfFolderExists(cmaname)
    savenametheta = rs.path + name + ".theta"
    copyfile(savenametheta, cmaname + name + ".theta")
    
    if(rs.regression=="RBFN"):
        savenamestruct = rs.path + name + ".struct"
        copyfile(savenamestruct, cmaname + name + ".struct")
        
def writeArray(numpyArray, foldername, name, extension):
    checkIfFolderExists(foldername)
    np.savetxt(foldername+name+extension, numpyArray)
    
