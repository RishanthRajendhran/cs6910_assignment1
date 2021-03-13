import numpy as np

def accuracy(y, pred):
    return (np.sum(np.equal(y, pred))/(len(y)))

def crossEntropyLoss(y, predProb):
    sum = 0
    for i in range(len(predProb)):
        sum = sum - (y[i]*np.log(predProb[i]))
    return sum

def MSEloss(y, pred):
    return (np.sum(np.subtract(y,pred)**2)/len(y))