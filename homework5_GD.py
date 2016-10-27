#!/usr/bin/env python

import sys
import numpy as np

def E(x):

    return ((x[0]*np.exp(x[1]))-(2*x[1]*np.exp(-x[0])))**2

def gradE(x):

    u = (2*np.exp(-2*x[0]))*((x[0]*np.exp(x[0]+x[1]))-(2*x[1]))*(np.exp(x[0]+x[1])+2*x[1])
    v = (2*np.exp(-2*x[0]))*(x[0]*np.exp(x[0]+x[1])-2)*(x[0]*np.exp(x[0]+x[1])-2*x[1])

    return np.array([u,v],dtype=np.float64)

def main():
    """Find the minimun"""
    eta = 0.1
    x0 = np.array([1.0,1.0],dtype=np.float64)
    #Gradient Descent Solution
    #Assign starting point
    x = x0
    iters = 0
    while True:
       x = x - eta*gradE(x)
       iters += 1
       if E(x) < 10e-14:
          break

    print "Gradient Descent"
    print "Error", E(x)
    print "Minimum point: ", x
    print "Iterations", iters

    #Coordinate Descent Solution
    #Assign starting point
    x = x0
    for i in range(15):
        x[0] = x[0] - eta*gradE(x)[0]
        x[1] = x[1] - eta*gradE(x)[1]

    print "Coordinate Descent"
    print "Error", E(x)
    print "Minimum point: ", x

    return 0

if __name__ == '__main__':
    
    print "Running HW5 problem 7"
    
    sys.exit(main())
