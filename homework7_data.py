import sys
import numpy as np
import pandas

inDatlen = 35*3
outDatlen = 250*3
k_val = 6

def loadDat():
    #Load data
    inDat = open('/Users/piris/Documents/edX/data/in.dta')
    outDat = open('/Users/piris/Documents/edX/data/out.dta')
    
    inDat2 = np.zeros(inDatlen)
    
    i = 0
    
    for line in inDat:
        for word in line.split():
            inDat2[i] = float(word)
            
            i += 1

    outDat2 = np.zeros(outDatlen)

    i = 0

    for line in outDat:
        for word in line.split():
            outDat2[i] = float(word)
            i += 1

    inDat2 = inDat2.reshape((35,3))
    outDat2 = outDat2.reshape((250,3))

    return inDat2, outDat2

def lr(D,y):
    
    lamb = 0.0
    
    X_ = D
    
    X_d = np.dot(np.linalg.inv(np.dot(X_.T,X_)+lamb*np.identity(len(X_.T))),X_.T)
    w = np.dot(X_d,y)

    return w

def transformation(X):
    
    #X = np.delete(X.T,2,0).T
    
    X_ = np.ones([len(X),int(k_val+1)])
    
    #Add first dimension
    i=0
    for x in X_:
        x[1:3] = X[i]
        x[3] = X[i][0]**2
        x[4] = X[i][1]**2
        x[5] = X[i][0]*X[i][1]
        x[6] = np.absolute(X[i][0]-X[i][1])
        #x[7] = np.absolute(X[i][0]+X[i][1])
        i += 1

    return X_


def evaluate(w,D,y):
    
    X_ = D
    
    Error = 0.0
    j = 0
    for x in X_:
        if np.sign(np.dot(w,x)) != y[j]:
            Error += 1
        j += 1

    return Error/len(D)

def split_data(X,y):

    train_x = X[0:25]
    train_y = y[0:25]
    valid_x = X[25:36]
    valid_y = y[25:36]

    return train_x,train_y,valid_x,valid_y

def main():
    """Implemented Solution"""

    inDat, outDat = loadDat()
    X = np.delete(inDat.T,2,0).T
    y =  inDat.T.take(2,axis=0)
    inDat_T = transformation(X)
    train_x,train_y,valid_x,valid_y = split_data(inDat_T,y)
    #print "inDat_T", inDat_T
    w = lr(train_x,train_y)
    Eval = evaluate(w,valid_x,valid_y)
    Xout = np.delete(outDat.T,2,0).T
    yout =  outDat.T.take(2,axis=0)
    Eout = evaluate(w,transformation(Xout),yout)
    print "K val: ", k_val
    print "Eval: ", Eval
    print "Eout: ", Eout
    
    print "Print Solution Q6: "
    X = np.ones([1000,2])
    for i in range(1000):
        X[i] = np.min(np.array([np.random.rand(),np.random.rand()]))
    
    print "e: ",X.mean()

    #print outDat
    #print "inDat: ",inDat
    #print "Ein: ",evaluate(w,inDat)/len(inDat)
    #print "Eout: ",evaluate(w,outDat)/len(outDat)

    return 0

if __name__ == '__main__':
    
    print "Running HW6 problem 2"
    
    sys.exit(main())
