#!/usr/bin/env python

import sys
import numpy as np
from numpy.linalg import inv, norm
from sklearn.cluster import KMeans
from sklearn import svm

pi = 3.1416

"""Heads = 1    Tails = 0"""

def gen_samples(N):
    """Return np array"""
    S = np.zeros((int(N),2))
    for i in range(len(S)):
        #Add dimenssion for bias
        S[i][0] = np.random.uniform(-1,1)
        S[i][1] = np.random.uniform(-1,1)
    
    return S

def target_function(X):
    """Return correct value"""
    Y = np.zeros((len(X),))
    index=0
    for x in X:
        Y[index] = np.sign(x[1]-x[0]+0.25*np.sin(pi*x[0]))
        index += 1
    
    return Y

def phi_mat(X,u,gamma):
    
    phi_mat = np.ones((len(X),len(u)+1))

    for k in range(len(u)):
        phi_mat[:,k] = np.exp(-gamma*norm(X-u[k],axis=-1))

    return phi_mat

def empty(U):
    var = False
    for u in U:
        if norm(u) == 0:
            var = True
    return var

def pseudo_inv(phi,Y):

    return np.dot(np.dot(inv(np.dot(phi.T,phi)),phi.T),Y)

def RBF_predict(X,w,b,u,gamma):
    Y_rbf = np.zeros(len(X))
    ans = []
    count = 0
    for x in X:
        ans = []
        for k in range(len(w)):
            ans.append(w[k]*np.exp(-gamma*norm(x-u[k]))+b)
        Y_rbf[count]=np.sign(sum(ans))
        count += 1

    return Y_rbf

def error(Y,Y_):
    er=0.0
    for i in range(len(Y)):
        if Y[i] != Y_[i]:
            er += 1
    return er/len(Y)

def Lloyd(X,K):
    U = gen_samples(K)
    while True:
        clusters = []
        for x in X:
            clusters.append(np.argmin([norm(x-u)for u in U]))
        oldu = U
        U = np.zeros((K,2))
        for i in range(K):
            count = 0.0
            newu = np.zeros((2,))
            for x,c in zip(X,clusters):
                if i == c:
                    count += 1
                    newu += x
            if count != 0:
                U[i] = newu/count
        if not norm(oldu-U):
            break

    return U

def main():
    """Run experiments."""
    Runs = 100
    N=100
    gamma = 1.5
    K = 12
    
    Ein_rbf = []
    Ein_svm = []
    Eout_rbf = []
    Eout_svm = []
    Ein_rbf_zero_count = 0.0
    Ein_svm_zero_count = 0.0
    Excluded = 0.0
    Kernel_better = 0.0
    
    for j in range(Runs):
        X = gen_samples(N)
        Y = target_function(X)
        #u = KMeans(n_clusters=K,init='random').fit(X).cluster_centers_
        u = Lloyd(X,K)
        if empty(u):
            Excluded += 1
            continue
        phi = phi_mat(X,u,gamma)
        w = pseudo_inv(phi,Y)
        b = w[-1]
        w = w[:len(w)-1]

        #RBF
        Y_RBF = RBF_predict(X,w,b,u,gamma)
        Ein_rbf.append(error(Y,Y_RBF))
        if error(Y,Y_RBF) == 0:
            Ein_rbf_zero_count += 1

        #SVM
        svm_class = svm.SVC(gamma=gamma)
        svm_class.fit(X,Y)
        Ein_svm.append(1.0-svm_class.score(X,Y))
        if (1.0-svm_class.score(X,Y)) == 0:
            Ein_svm_zero_count += 1
        
        #Eout
        X_ = gen_samples(N*10)
        Y_ = target_function(X_)
        Eout_rbf.append(error(Y_,RBF_predict(X_,w,b,u,gamma)))
        Eout_svm.append(error(Y_,svm_class.predict(X_)))
        if error(Y_,RBF_predict(X_,w,b,u,gamma)) > error(Y_,svm_class.predict(X_)):
            Kernel_better += 1

    print "Ein_rbf: ",np.mean(Ein_rbf)
    print "Ein_rbf_zero_count: ",Ein_rbf_zero_count/(Runs-Excluded)
    print "Ein_svm: ",np.mean(Ein_svm)
    print "Ein_svm_zero_count: ",Ein_svm_zero_count/(Runs-Excluded)
    print "Eout_rbf: ",np.mean(Eout_rbf)
    print "Eout_svm: ",np.mean(Eout_svm)
    print "Kernel_better: ",Kernel_better/(Runs-Excluded)
    print "Excluded: ",Excluded
    
    return 0

if __name__ == '__main__':
    
    print "Running HW2 Linear"
    
    sys.exit(main())
