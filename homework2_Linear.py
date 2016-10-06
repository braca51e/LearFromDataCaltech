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

def target_function():
    """Return the values of w"""
    f = np.zeros(3)
    
    point1 = gen_samples(1)
    point2 = gen_samples(1)
    x1 = point1[0][1]
    y1 = point1[0][2]
    x2 = point2[0][1]
    y2 = point2[0][2]
    
    #Generate equation line
    a = ((y2-y1)/(x2-x1)) #a
    b = -1               #b
    c = y1 - a*x1        #c
    
    f[0] = c
    f[1] = a
    f[2] = b
    
    return f

def flip_sign(X):
    F = np.ones(int(len(X)*0.1))
    for k in range(int(len(X)*0.1)):
        F[k]=int(np.random.uniform(0,len(X)))
    
    for f in F:
        X[int(f)][3] *= -1
        #X[int(f)][0] = 1

def evaluate_sample(X,f):
    """Assign label to each sample"""
    X_ = np.zeros((len(X),4))
    for i in range(len(X)):
        #Compute as stated in the lecture
        y = np.sign(np.dot(f,X[i]))
        X_[i]=np.append(X[i],y)

    return X_

def evaluate_nonlinear(X):
    """Assign label to each sample"""
    X_ = np.zeros((len(X),4))
    for i in range(len(X)):
        #Compute as stated in the lecture
        y = np.sign(np.square(X[i]).sum()-0.6)
        X_[i]=np.append(X[i],y)
    
    return X_

def get_labels(X,f):
    """Assign label to each sample"""
    y = np.zeros(len(X))
    for i in range(len(X)):
        #Compute as stated in the lecture
        y[i] = np.sign(np.dot(f,X[i]))
    
    return y

def eq_line_plot(a,c):
    x = np.arange(-1.0,1,0.01)
    y = a*x + c
    plt.plot(x,y)
    plt.xlim(-1,1)
    plt.ylim(-1,1)

def plot_samples(X):
    for i in range(len(X)):
        if X[i][3] == -1:
            plt.plot(X[i][1],X[i][2],'bs')
        else:
            plt.plot(X[i][1],X[i][2],'ro')

def pla_algo(E=None,N=None,X=None,w=None,f=None):
    
    #Generate correct labels
    X_ = evaluate_sample(X,f)
    #All points initially misclassified
    missPoints = X_
    #Initial w values (learned function LR)
    w_ = w
    #iterations
    iters = 0.0
    while len(missPoints) != 0:
        iters += 1
        #Pick a random point
        pick = np.random.randint(len(missPoints))
        x_ = missPoints[pick]
        #Update weight
        w_ = w_ + np.multiply(x_[3],x_[0:3])
        #Reclassify points
        missPoints = []
        for x in X_:
            if np.sign(np.dot(w_,x[0:3])) != np.sign(x[3]):
                missPoints.append(x)

    return iters

def linear_regression(N=None,X=None,f=None):
    Ein = 0.0
    Eout = 0.0
    gfuns = 0.0
    
    #Generate correct labels
    y = get_labels(X,f)
    #Learned function
    X_dag = np.dot(inv(np.dot(X.T,X)),X.T)
    g = np.dot(X_dag,y)
    #Generate LR labels
    y_ = get_labels(X,g)
    #Compute error
    wrongClass = 0
    for j in range(len(y)):
        if y[j] != y_[j]:
            wrongClass += 1
    
    Ein = wrongClass/float(N)
    #Error out of sample computation
    S = gen_samples(1000)
    #Generate correct labels
    y = get_labels(S,f)
    #Generate LR labels
    y_ = get_labels(S,g)
    #Compute error
    wrongClass = 0
    for j in range(len(y)):
        if y[j] != y_[j]:
            wrongClass += 1
    Eout = wrongClass/1000.0

    return Ein,Eout,g

def main():
    """Run experiments."""
    E = 1000
    N=10
    
    Ein = np.zeros(E)
    Eout = np.zeros(E)
    gfuns = np.zeros((E,3))
    iterations = np.zeros(E)
    
    for j in range(E):
        #Get samples
        X = gen_samples(N)
        #Generate target function
        f = target_function()
        #Problem 5, 6 and 7
        #Run Linear Regression
        Ein[j],Eout[j],gfuns[j] = linear_regression(N,X,f)
        ##Run PLA
        iterations[j] = pla_algo(E,N,X,gfuns[j],f)
    
    print "....Linear Regression...."
    print "Ein: ", Ein.mean()
    print "Eout: ", Eout.mean()
    print "g: ", gfuns.mean()
    print "....Linear Perceptron Algorithm...."
    print "Iterations: ", iterations.mean()
    
    return 0

if __name__ == '__main__':
    
    print "Running HW2 Linear"
    
    sys.exit(main())
