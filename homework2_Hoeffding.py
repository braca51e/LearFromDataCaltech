#!/usr/bin/env python

import sys
import numpy as np
#import matplotlib.pyplot as plt

"""Heads = 1    Tails = 0"""

def gen_samples(N):
    """Return np array"""
    S = np.zeros((int(N),3))
    for i in range(len(S)):
        #Add dimenssion for bias
        S[i][0] = 1
        S[i][1] = np.random.uniform(-1,1)
        S[i][2] = np.random.uniform(-1,1)
    
    return S

def coin_flip(coins,flips):
    """Return simulation array"""
    #index of min heads
    heads = 10
    S = np.zeros((int(coins),int(flips)))
    
    for j in range(len(S)):
        S[j] = np.random.randint(2, size=flips)
        #assign cmin
        if heads > int(S[j].sum()):
            heads = int(S[j].sum())
            cmin = j

    #Get random coin
    crand = int(np.random.uniform(0,coins-1))

    return cmin,crand,S

def main():
    """Run experiments."""
    
    #Experiments
    E = 1000
    #Coins
    C = 1000
    #Flips
    F = 10
    #Index c1
    c1 = 0
    #v values
    v1 = np.zeros((E))
    vrand = np.zeros((E))
    vmin = np.zeros((E))
    
    for i in range(E):
        cmin,crand,S = coin_flip(C,F)
        v1[i] = float(S[c1].sum())/10.0
        vrand[i] = float(S[crand].sum())/10.0
        vmin[i] = float(S[cmin].sum())/10.0
    
    
    print "Average vmin: ",vmin.mean()
    print "Average vrand: ",vrand.mean()
    print "Average v1: ",v1.mean()

    return 0

if __name__ == '__main__':
    
    print "Running HW2"
    
    sys.exit(main())
