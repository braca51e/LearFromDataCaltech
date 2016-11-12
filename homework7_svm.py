#!/usr/bin/env python

import sys
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp

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

def evaluate_sample(X,f):
    """Assign label to each sample"""
    X_ = np.zeros((len(X),4))
    for i in range(len(X)):
        #Compute as stated in the lecture
        y = np.sign(np.dot(f,X[i]))
        X_[i]=np.append(X[i],y)

    return X_

def get_labels(X,f):
    """Assign label to each sample"""
    y = np.zeros(len(X))
    for i in range(len(X)):
        #Compute as stated in the lecture
        y[i] = np.sign(np.dot(f,X[i]))
    
    return y

def pla_algo(X=None,f=None):
    
    #Generate correct labels
    X_ = evaluate_sample(X,f)
    #All points initially misclassified
    missPoints = X_
    #Initial w values
    w = np.array([0.0,0.0,0.0])
    #iterations
    iters = 0.0
    while len(missPoints) != 0:
        iters += 1
        #Pick a random point
        pick = np.random.randint(len(missPoints))
        x_ = missPoints[pick]
        #Update weight
        w = w + x_[3]*x_[0:3]
        #Reclassify points
        missPoints = []
        for x in X_:
            if np.sign(np.dot(w,x[0:3])) != np.sign(x[3]):
                missPoints.append(x)

    return w

def svm_algo(X=None,y=None):

    Nsamples, Nsample_size = X.shape

    Kernel = np.zeros((Nsamples,Nsamples))
    for i in range(Nsamples):
        for j in range(Nsamples):
            Kernel[i,j] = np.dot(X[i],X[j])

    P = matrix(np.outer(y,y)*Kernel)
    q = matrix(np.ones(Nsamples)*-1.0)
    A = matrix(y,(1,Nsamples))
    b = matrix(0.0)
    G = matrix(np.diag(np.ones(Nsamples)*-1.0))
    h = matrix(np.zeros(Nsamples))

    sol = qp(P,q,G,h,A,b)

    alpha = sol['x']

    w = np.array([0.0,0.0])

    count_sv = 0.0
    for i in range(Nsamples):
        if alpha[i] > 1e-6:
            w += alpha[i]*y[i]*X[i]
            count_sv += 1
            alpha_b = i

    #print "alpha: ",alpha

    b = (1 - y[alpha_b]*np.dot(w,X[alpha_b]))/(y[alpha_b])

    return w,b,count_sv

def main():
    """Run experiments."""
    E = 1000
    N=100
    
    Better_svm = 0.0
    Disccount = 0.0
    N_sv= []
    
    for j in range(E):
        #Get samples
        X = gen_samples(N)
        Xeval = gen_samples(N*100)
        #Generate target function
        f = target_function()
        #Get labels
        y = get_labels(X,f)
        #Check not all in one side
        if len(y) == np.abs(y.sum()):
            Disccount += 1
            continue

        Gpla =  pla_algo(X,f)
        #Measure dissagreement
        DisGpla = 0.0
        for x in Xeval:
            if np.sign(np.dot(f,x)) != np.sign(np.dot(Gpla,x)):
                DisGpla += 1

        DisGpla = DisGpla/(N*100)

        Gsvm,bsvm,count_sv = svm_algo(np.delete(X.T,0,0).T,y)
        N_sv.append(count_sv)
        #Measure disagreement
        Dissvm = 0.0
        for x in Xeval:
            if np.sign(np.dot(f,x)) != np.sign(np.dot(Gsvm,x[1:3])+bsvm):
                Dissvm += 1
        
        Dissvm = Dissvm/(N*100)
        
        if DisGpla > Dissvm:
            Better_svm += 1

    print "N_sv: ", np.mean(N_sv)
    print "Better_svm: ", Better_svm/(float(E)-Disccount)
                        
    
    return 0

if __name__ == '__main__':
    
    print "Running HW7 Linear"
    
    sys.exit(main())
