#!/usr/bin/env python

import sys
import numpy as np
from numpy.linalg import inv, norm
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

def flip_sign(X):
    #F = np.ones(int(len(X)*0.1))
    for k in range(int(len(X)*0.1)):
        #F[k]=int(np.random.uniform(0,len(X)))
        t = int(np.random.uniform(0,len(X)))
        X[t][3] *= -1
    
    #for f in F:
        #X[int(f)][3] *= -1
        #X[int(f)][0] = 1

def evaluate_nonlinear(X):
    """Assign label to each sample"""
    X_ = np.zeros((len(X),4))
    for i in range(len(X)):
        #Compute as stated in the lecture
        y = np.square(X[i][1])+np.square(X[i][2])-0.6
        y = np.sign(y)
        X_[i]=np.append(X[i],y)
    
    return X_

def get_labels1(X):
    """Assign label to each sample"""
    y = np.zeros(len(X))
    for i in range(len(X)):
        #Compute as stated in the lecture
        y[i] = X[i][3]
    
    return y

def nonlinear_transform(X):
    """Assign label to each sample"""
    X_ = np.zeros((len(X),6))
    for i in range(len(X)):
        #Compute transform
        x1 = X[i][1]
        x2 = X[i][2]
        x1_x2 = X[i][1]*X[i][2]
        x1_2 = X[i][1]*X[i][1]
        x2_2 = X[i][2]*X[i][2]
        X_[i] = np.array([1,x1,x2,x1_x2,x1_2,x2_2])
    
    return X_

def get_labels2(X,f):
    """Assign label to each sample"""
    y = np.zeros(len(X))
    for i in range(len(X)):
        #Compute as stated in the lecture
        y[i] = np.sign(np.dot(f,X[i]))
    
    return y

def linear_regression(E=None,N=None):
    Ein = np.zeros(E)
    Eout = np.zeros(E)
    gfuns = np.zeros((E,3))
    w_s = np.zeros((10,6))
    
    for i in range(E):
        #Get samples
        X = gen_samples(N)
        X_ = evaluate_nonlinear(X)
        #Generate noiseless labels
        y = get_labels1(X_)
        flip_sign(X_)
        #Generate noisy labels
        y_ = get_labels1(X_)
        #Learned function
        X_dag = np.dot(inv(np.dot(X.T,X)),X.T)
        g = np.dot(X_dag,y_)
        #Generate LR noisy labels
        y_r = get_labels2(X,g)
        #Compute error
        wrongClass = 0
        for j in range(len(y)):
            if y[j] != y_r[j]:
                wrongClass += 1
        
        Ein[i] = wrongClass/float(N)
        gfuns[i] = g

        #Compute transform
        if i < len(w_s):
            X_t = nonlinear_transform(X)
            X_tdag = np.dot(inv(np.dot(X_t.T,X_t)),X_t.T)
            w_ = np.dot(X_tdag,y_)
            w_s[i] = w_

        #Eout Computation
        Xt = gen_samples(N)
        X_t = evaluate_nonlinear(Xt)
        yt = get_labels1(X_t)
        flip_sign(X_t)
        yt_n = get_labels1(X_t)
        Xt_ = nonlinear_transform(Xt)
        yt_r = get_labels2(Xt_,w_)
            
        wrongClass = 0
        for j in range(len(y)):
            if yt_n[j] != yt_r[j]:
                wrongClass += 1
        Eout[i] = wrongClass/float(N)

    return Ein.mean(),Eout.mean(),g,w_s.mean(axis=0)

def main():
    """Run experiments."""
    E = 1000
    N=1000
    ###Problem 8
    Ein,Eout,g,w_ = linear_regression(E,N)
    print "....Linear Regression...."
    print "Ein: ", Ein
    print "Eout: ", Eout
    print "g: ", g
    print "w_",w_
    
    return 0

if __name__ == '__main__':
    
    print "Running HW2 Linear"
    
    sys.exit(main())
