# -*- coding: utf-8 -*-

import sys
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def load_data():
    #Load data
    train_data = open('./data/features.train')
    test_data  = open('./data/features.test')
    
    train_data_len = sum(1 for line in open('./data/features.train'))
    test_data_len  = sum(1 for line in open('./data/features.train'))
    
    train_data2 = np.zeros(train_data_len*3)
    
    i = 0
    
    for line in train_data:
        for word in line.split():
            train_data2[i] = float(word)
            i += 1

    test_data2 = np.zeros(test_data_len*3)

    i = 0

    for line in test_data:
        for word in line.split():
            test_data2[i] = float(word)
            i += 1

    print "train_data2: ", type(train_data2)    
    
    train_data2 = train_data2.reshape((train_data_len,3))
    test_data2   = test_data2.reshape((test_data_len,3))

    return train_data2, test_data2

def split_data(X):

    data_x = np.delete(X.T,0,0).T
    data_y = X.T[0]

    return data_x,data_y
    
def NvsAll(n,Y):
    
    new_y = np.zeros(len(Y))
    index = 0
    
    for y in Y:
        if y == n:
            new_y[index] = 1.0
        else:
            new_y[index] = -1.0
            
        index += 1
            
    return new_y
    
def NvsM(n,m,X,Y):
    
    new_x = []
    new_y = []
    index = 0
    
    for y in Y:
        if y == n:
            new_y.append(1.0)
            new_x.append(X[index])
        elif y == m:
            new_y.append(-1.0)
            new_x.append(X[index])
            
        index += 1 
        
    return np.array(new_x),np.array(new_y)
           
def main():
    """Implemented Solution"""
    digit_num = 10
    Ein_ovo = 0.0
    Eout_ovo = 0.0
    Ein_ovr = np.zeros((digit_num,))
    Runs = 100
    
    train_data, test_data = load_data()
    
    train_x, train_y = split_data(train_data)
    test_x, test_y = split_data(test_data)
    
    #one vs all section
    print "Answers 2-6"
    print "one vs all section"
    svm_class = svm.SVC(C=0.01,degree=2,kernel='poly', coef0=1.0)   
    
    for i in range(10):
        
        Y = NvsAll(i,train_y)
        svm_class.fit(train_x,Y)
        #In-saple error
        Ein_ovr[i] = svm_class.score(train_x,Y)
        #Get number of support vectors
        print "Number: ",i
        print "support_vectors", len(svm_class.support_vectors_)
        
    print "Ein",1.0-Ein_ovr        
    print "max Ein: ", np.argmax(Ein_ovr)
    print "min Ein: ", np.argmin(Ein_ovr)
    
    #CV 
    print "Answers 7-8"
    N = 1
    M = 5
    n_splits = 10
    C = ['0.0001','0.001','0.01','0.1','1.0']
    C_selesct = np.zeros((Runs,))
    Ecv = np.zeros((Runs,))
    new_x,new_y = NvsM(N,M,train_x,train_y)
    
    for j in range(Runs):
        kf = KFold(n_splits=n_splits,shuffle=True)
        
        c_i = 0
        Ec = np.zeros((len(C),))
        for c in C:
            Esplit = np.zeros((n_splits,))
            svm_class_cv = svm.SVC(C=float(c),degree=2,kernel='poly', coef0=1.0)  
        
            split_i = 0
            for train_index, test_index in kf.split(new_x,new_y):
                svm_class_cv.fit(new_x[train_index],new_y[train_index])
                Esplit[split_i] = svm_class_cv.score(new_x[test_index],new_y[test_index])
                split_i += 1
                
            Ec[c_i] = Esplit.mean()
            c_i += 1 

        C_selesct[j] = Ec.argmax()
        Ecv[j] = 1.0-Esplit.mean()
    
    #0=a,b=1,c=2 ..................
    print "C=0.0001",(C_selesct == 0).sum()
    print "C=0.001",(C_selesct == 1).sum()
    print "C=0.01",(C_selesct == 2).sum()
    print "C=0.1",(C_selesct == 3).sum()
    print "C=1",(C_selesct == 4).sum()    
    print "Ecv: ",Ecv.mean()
    #one vs one
    print "Answers 9-6"
    print "one vs one"
    N = 1
    M = 5
    svm_class2 = svm.SVC(C=10e6,degree=2,kernel='rbf', coef0=1.0,decision_function_shape='ovo')
    new_x,new_y = NvsM(N,M,train_x,train_y)
    new_test_x,new_test_y = NvsM(N,M,test_x,test_y)
    svm_class2.fit(new_x,new_y)
    Ein_ovo = svm_class2.score(new_x,new_y)
    Eout_ovo = svm_class2.score(new_test_x,new_test_y)
    print "C=10e6"
    print "support_vectors", len(svm_class2.support_vectors_)
    print "Ein:",1.0-Ein_ovo
    print "Eout:",1.0-Eout_ovo
    
    return 0

if __name__ == '__main__':
    
    print "Running HW8 problem 2-10"
    
    sys.exit(main())
