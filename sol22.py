# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:10:29 2017

@author: Dingdang
"""

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


iris_data = np.loadtxt(r'C:\Users\Dingdang\Google Drive\CS\Dal courses\Term4\ML\Assignment\4\ML_Assignment4\fisheriris.data', delimiter=',')

Y = np.reshape(iris_data.T[-1], (1, len(iris_data)))
Y = Y.astype(int)

X = np.reshape(iris_data.T[:-1], (len(iris_data[0]) - 1, len(iris_data)))

def calculateProb(phi, X, u, co_var):
    phi = np.reshape(phi, (1,1)) #1 * 1
    X = np.reshape(X, (1, X.shape[0]))  # 1 * 4
    u = np.reshape(u, (1, u.shape[0]))  # 1 * 4
    
    qResult = phi * (np.power(np.e, (-0.5 * (X - u) @ np.linalg.inv(co_var) @ (X - u).T))) /\
    (np.power(2*np.pi, X.shape[0]/2) * np.sqrt(np.linalg.det(co_var)))

    return qResult[0]
    
#calcProbOfClassGivenX is used for the E step in EM algorithm
def calcProbOfClassGivenX (X, PHI, U, CO_VAR, j, k):
    denominator = 0
    numerator = calculateProb(PHI[j], X, U[j].T, CO_VAR[j]) * PHI[j]
    for i in range(k):
        denominator += calculateProb(PHI[i], X, U[i].T, CO_VAR[i]) * PHI[i]
    
    return numerator/denominator

#k is num of classes and epsilon is the convergence threshold
def em(X, k, epsilon): 
    
    #initialize MUs for each of the classes by randomly choose k samples from X
    #as the centres for k classes
    Us = np.array([np.ones(X.shape[0]) for _ in range(k)])

    for i in range(k):
        Us[i] = X.T[np.random.randint(0, X.shape[1])]

    PHI = np.array([[1/k] for _ in range(k)])
    CO_VARs = np.array([np.identity(X.shape[0]) for _ in range(k)])
    
    #keep track of E step values
    classQs = np.array([np.zeros(X.shape[1]) for _ in range(k)])
    
    converge = 1000
    iterNum = 0
    
    try:   
        while(converge > epsilon):
            #keep track of E step values
            #classQs = np.array([np.zeros(X.shape[1]) for _ in range(k)])
            
            #make a copy of the MUs in the previous step
            U_pre = np.copy(Us)
    
            #E step#       
            for i in range(X.T.shape[0]):
                for j in range(k):
                    classQs[j][i] = calcProbOfClassGivenX(X.T[i], PHI, Us, CO_VARs, j, k)
            
            #M step#       
            for cInx in range(k):
                try:
                    qTemp = np.reshape(classQs[cInx],(len(classQs[cInx]), 1)) #n * 1
                    qSum = np.sum(qTemp)
                    if math.isnan(qSum):
                        print("nanananan", math.isnan(qSum))
                        break
                    print(cInx, "class:", qSum)
            
                    Us[cInx] = np.sum(X.T * qTemp, axis = 0)/qSum #Us 1*4
                    CO_VARs[cInx] = (qTemp.T * (X.T - Us[cInx]).T) @ (X.T - Us[cInx])/qSum
                    PHI[cInx] = qSum/len(classQs[cInx])
                except:
                    break
                
            print("Us:", Us)        
            print("iteration:", iterNum)
            iterNum += 1
            
            #check convergence based on the Euclidean distance between the updated
            #MUs and the previous MUs
            converge = np.sum(np.square(Us - U_pre))
            print("converge:", converge)
    except:
        print("Except Us:", Us)
        
em(X, 3, 0.01)    
"""        
    classSamples = list([] for _ in range(k))
    classQs = list([] for _ in range(k))
#        for i in range(k):
#            preIterCost += classQs[i] @ np.log(calculateProb(PHI[i], X[i], U[i].T, CO_VAR[i]) * PHI[i] / classQs[i]).T
#        #E step#
    
    for i in range(X.T.shape[0]):
        maxProb = 0
        classLabel = -1
        for j in range(k):
            #classQs[j][i] = calcProbOfClassGivenX(X.T[i], PHI, Us, CO_VARs, j, k)
            
            tempProb = calcProbOfClassGivenX(X.T[i], PHI, Us, CO_VARs, j, k)
            if tempProb > maxProb:
                classLabel = j
                maxProb = tempProb
        classQs[classLabel].append(tempProb)
        classSamples[classLabel].append(X.T[i])
"""       
em(X, 3, 0.01)    


    
a = np.array([[1,2],[3,4], [5,6]])
z = np.array([5, 1])
x = np.copy(a)


b = np.zeros(np.shape(a))
c = []
c.append(a)
c.append(a)
c.append(a)
c = np.array(c)
d = np.array([1] * 3)
e = np.cov(X)
f = np.array([1, 2, 3, 4])
g = np.exp(f)
h = np.array([np.ones(X.shape[0]) for _ in range(3)])
for i in range(3):
    h[i] += 2 * i
i = [np.identity(X.shape[0])] * 3
classes = [[] for _ in range(3)] 
classes[0].append([1,2,3])

CO_VARs = np.array([np.identity(X.shape[0]) for _ in range(3)])

y = [1,1]
np.reshape(y,(len(y), 1))
PHI = np.array([[1/2]  for _ in range(3)])

classeQs = np.array([np.ones(X.shape[1]) for _ in range(3)])

Us = np.array([np.ones(X.shape[0]) for _ in range(3)])

classSamples = np.array([[] for _ in range(3)])
classSamples = list([] for _ in range(3))

Us = np.array([np.ones(X.shape[0]) for _ in range(3)])

for i in range(3):
    Us[i] = X.T[np.random.randint(0, X.shape[1])]

costInnerLogtest = np.array(np.zeros(X.shape[1]))
Xsuffle = np.random.shuffle(X.T)

PHItest = np.array([np.random.rand() for _ in range(3)])