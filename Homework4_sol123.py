import sys
import numpy as np
#import matplotlib.pyplot as plt

"""Heads = 1    Tails = 0"""

def sample(N):
    """Return np array"""
    X = np.zeros(N)
    
    for i in range(N):
        X[i] = np.random.uniform(-1,1)
    
    #print "X value: ",X
    
    return X

def f(S):
    """Evalute input values"""
    Y = np.zeros(len(S))
    for i in range(len(S)):
        Y[i] = np.sin(3.1416*S[i])
    
    #print "Print Y:",Y
    return Y

def solve(X,Y):
    """Find a value"""
    A = np.zeros(1)
    
    A = np.dot(X.T,Y)/np.dot(X.T,X)
    
    X_2 = np.square(X)
    A_2 = np.dot(X_2.T,Y)/np.dot(X_2.T,X_2)
    
    return A,A_2

def main():
    """Run experiments."""
    #Number of experiments
    E = 100000
    Avals = np.zeros(E)
    Avals_2 = np.zeros(E)
    #A_vals = np.zeros(E)
    Bvals = np.zeros(E)
    Bvals_2 = np.zeros(E)
    Vvals = np.zeros(E)
    Vvals_2 = np.zeros(E)
    
    for i in range(E):
        X = sample(2)
        Y = f(X)
        Avals[i], Avals_2[i] = solve(X,Y)
    
    a = Avals.mean()
    a_2 = Avals_2.mean()
    #Find bias value
    for i in range(E):
        X = sample(1)
        Y = f(X)
        Bvals[i] = (a*X-Y)*(a*X-Y)
        Bvals_2[i] = (a_2*X-Y)*(a_2*X-Y)

    #Find variance value
    for i in range(E):
        X = sample(2)
        Y = f(X)
        a_d,a_d2 = solve(X,Y)
        X = sample(1)
        
        g_d = a_d*X
        g = a*X
        g_d2 = a_d2*X
        g2 = a_2*X
        
        Vvals[i] = (g_d-g)*(g_d-g)
        Vvals_2[i] = (g_d-g2)*(g_d-g2)
    

    print "Linear Solution"
    print "Mean value a: ",a
    print "Bias value a: ",Bvals.mean()
    print "Var value a: ",Vvals.mean()
    print "Quadratic Solution"
    print "Mean value a: ",a_2
    print "Bias value a: ",Bvals_2.mean()
    print "Var value a: ",Vvals_2.mean()
    return 0

if __name__ == '__main__':
    
    print "Running HW4"
    
    sys.exit(main())
