import sys
import numpy as np

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
    """Assign label to each sample 4th dimension is y"""
    X_ = np.zeros((len(X),4))
    for i in range(len(X)):
        #Compute as stated in the lecture
        y = np.sign(np.dot(f,X[i]))
        X_[i]=np.append(X[i],y)
    
    return X_

def cross_entropy(x,w):

    return np.log(1+np.exp(-x[3]*np.dot(w.T,x[0:3])))

def cross_entropy_grad(x,w):
    
    return -(x[3]*x[0:3])/(np.exp(x[3]*np.dot(w.T,x[0:3]))+1)

def sigmoid(x,w):

    return (1.0)/(1+np.exp(-np.dot(w.T,x[0:3])))

def main():
    """Implemented Logistic Regression"""
    eta = 0.01
    Exp = 100
    N = 100
    Epochs = np.zeros([100,1])
    Eout = np.zeros([100,1])
    for i in range(Exp):
        X = gen_samples(N)
        f = target_function()
        X_ = evaluate_sample(X,f)
        #minimize step
        w = np.array([0.0,0.0,0.0])
        epochs = 0
        while True:
            w_t = w
            for x in X_:
                w = w - eta*cross_entropy_grad(x,w)
            epochs += 1
            if np.linalg.norm(w_t-w) < 0.01:
                break

        Epochs[i] = epochs

        X = evaluate_sample(gen_samples(100),f)
        Erun = np.zeros([100,1])
        index = 0
        for x in X:
            Eout[i] = cross_entropy(x,w)

    print "Epochs: ",Epochs.mean()
    print "Eout: ",Eout.mean()

    return 0

if __name__ == '__main__':
    
    print "Running HW5 problem 8-9"
    
    sys.exit(main())
