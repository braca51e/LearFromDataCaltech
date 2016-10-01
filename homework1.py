#!/usr/bin/env python

import sys
import numpy as np
#import matplotlib.pyplot as plt

def gen_samples(N):
    arr = []
    for i in range(N):
        #Add dimenssion for bias
        x0 = 1
        x1 = np.random.uniform(-1,1)
        x2 = np.random.uniform(-1,1)
        arr.append([x0,x1,x2])
    return arr

def target_function():
    """Return the values of w"""
    point1 = gen_samples(1)
    point2 = gen_samples(1)
    x1 = point1[0][1]
    y1 = point1[0][2]
    x2 = point2[0][1]
    y2 = point2[0][2]
    #Generate equation line
    a =((y2-y1)/(x2-x1))
    b = -1
    c = y1 - a*x1

    return [c,a,b]

def evaluate_sample(X,w):
    for i in range(len(X)):
        #Compute as stated in the lecture
        y = np.sign(np.dot(w,X[i]))
        X[i].append(y)

def get_laebel(X,w):
    y = []
    for i in range(len(X)):
        #Compute as stated in the lecture
        y.append(np.sign(np.dot(w,X[i])))

    return y


def eq_line_plot(a,c):
    x = np.arange(-1.0,1,0.01)
    y = a*x + c
    plt.plot(x,y)

def plot_samples(X):
    for x in X:
        if x[3] == -1:
            plt.plot(x[1],x[2],'bs')
        else:
            plt.plot(x[1],x[2],'ro')

def hw1_solve(R,N):

    g = []
    iter = []
    
    while R > 0:
        #Get samples
        X = gen_samples(N)
        #Generate target function
        w = target_function()
        #Generate correct labels
        evaluate_sample(X,w)
        #Test plot
        #eq_line_plot(w[1],w[0])
        #plot_samples(X)
        #plt.ylim([-1,1])
        #plt.xlim([-1,1])
        #plt.show()
        #All points initially misclassified
        missPoints = X
        #Initial w values (learned function)
        w_ = [0.0,0.0,0.0]
        #Actual Implementation of PLA
        #Count number of iters
        count = 0
        while len(missPoints) != 0:
            count += 1
            #Pick a random point
            pick = np.random.randint(len(missPoints))
            x_ = missPoints[pick]
            #Update weight
            w_ = w_ + np.multiply(x_[3],x_[0:3])
            #Reclassify points
            missPoints = []
            for x in X:
                if np.sign(np.dot(w_,x[0:3])) != np.sign(x[3]):
                    missPoints.append(x)
        R -= 1
        iter.append(count)
        #Find probability of disagreement
        X1 = gen_samples(1000)
        y = get_laebel(X1,w)
        #print y
        #print "%%%%%%%%"
        y_ = get_laebel(X1,w_)
        #print y_
        missClass = 0
        for i in range(len(y)):
            if y[i] != y_[i]:
                missClass += 1
        
        g.append(missClass/1000.0)
                               
    return np.mean(g),np.mean(iter)

def main():
    """Run experiments."""
    Runs = 1000
    N = 100
    g,iter =hw1_solve(Runs,N)
    print "Probability of g disagreeement:", g
    print "iters: ", iter
    
    return 0

if __name__ == '__main__':
    
    print "Running HW1"
    
    sys.exit(main())
