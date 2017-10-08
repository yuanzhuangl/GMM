#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement EM algrithem base on iris data set

Created on Mon Oct 7 13:58:01 2017

@author: Yuan
"""

from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class EM_Object(object):

    def __init__(self,files,k):
        self.data_file = files[0]
        self.feature_file = files[1]
        self.class_set = []
        self.avg = []

        # save sample data as matrix
        self.X = empty((0))

        # save number of cols and rows
        self.sample_count = 0
        self.feature_count = 0


        # parameters of GMM
        self.k = k
        self.init_cov = []
        self.init_mu = []
        self.init_phi = []

        # output result
        self.result = pd.DataFrame(columns=['sepal length', 'sepal width',
                                            'petal length', 'petal width','class', 'Prob'])
        self.centroids = []

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # load data as numpy array
    def load_data(self):
        data_set = loadtxt(self.data_file,delimiter=',')
        self.X = mat(data_set.T[:-1].T)
        self.sample_count, self.feature_count = shape(self.X)
        self.avg = [average(col) for col in data_set.T[:-1]]

        with open(self.feature_file,'r') as f:
            self.class_set = f.readline().split(',')

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # init parameters
    def init_par(self):
        self.load_data()
        
        # init phi distribution
        for i in range(self.k):
            self.init_phi.append(1 / self.k)

        # randomly select k sample as initial mu of k class
        self.init_mu = [self.X[i, :] for i in random.randint(0,150,size=self.k)]

        # init identity matrix as initial covariance matrix
        self.init_cov = [mat(np.identity(self.feature_count)) for _ in range(self.k)]

#calculate multivariate Normal distribution probabilities of each sample(gamma)
def G_prob(x, mu, cov):
    n = len(x[0])
    e_power = float(-0.5 * (x - mu) * (cov.I) * ((x - mu).T))
    Deno = power(2 * pi, n / 2) * power(linalg.det(cov), 0.5)
    gamma = power(e, e_power) / Deno
    return gamma


#implement EM Algorithm
def EM(EM_Object):
    # init parameters
    phi = EM_Object.init_phi
    cov = EM_Object.init_cov
    mu = EM_Object.init_mu
    k = EM_Object.k

    # init probabilities set
    gamma = mat(zeros((EM_Object.sample_count, k)))

    # Start Iteration
    dif = 1
    threshold = 1e-3
    while dif > threshold:
        mu_pre = [item for item in mu]
        # step E
        for j in range(EM_Object.sample_count):
            px = 0
            for i in range(k):
                gamma[j, i] = phi[i] * G_prob(EM_Object.X[j, :], mu[i], cov[i])
                px += gamma[j, i]
            for i in range(k):
                gamma[j, i] /= px
        sum_gamma = sum(gamma, axis=0)

        # step M
        for i in range(k):
            mu[i] = mat(zeros((1, EM_Object.feature_count)))
            cov[i] = mat(zeros((EM_Object.feature_count, EM_Object.feature_count)))
            for j in range(EM_Object.sample_count):
                mu[i] += gamma[j, i] * EM_Object.X[j, :]
            mu[i] /= sum_gamma[0, i]
            for j in range(EM_Object.sample_count):
                cov[i] += gamma[j, i] * (EM_Object.X[j, :] - mu[i]).T * (EM_Object.X[j, :] - mu[i])
            cov[i] /= sum_gamma[0, i]
            phi[i] = sum_gamma[0, i] / EM_Object.sample_count

        # check whether mu are convergence
        dif = 0
        for i in range(k):
            distance = (mu[i]-mu_pre[i])*(mu[i]-mu_pre[i]).T
            dif += distance[0,0]
    return gamma


def cluster(EM_Object):
    # init centroids set for different classes
    classification = mat(zeros((EM_Object.sample_count, 2)))
    gamma = EM(EM_Object)


    for i in range(EM_Object.sample_count):
        # Align to groups (return the index of biggest probability, and such prob)
        classification[i, :] = argmax(gamma[i, :]), amax(gamma[i, :])
        temp = [item for item in squeeze(np.asarray(EM_Object.X[i, :]))] + [argmax(gamma[i, :]), amax(gamma[i, :])]
        EM_Object.result.loc[i] = temp

        # update centroids
    for j in range(EM_Object.k):
        pointsInCluster = EM_Object.X[nonzero(classification[:, 0].A == j)[0]]
        EM_Object.centroids.append(mean(pointsInCluster, axis=0))

    # set 'class' column data type to int
    EM_Object.result['class'] = pd.to_numeric(EM_Object.result['class'],downcast='signed', errors='coerce')



def plot(EM_Object):

    # set dot type and color
    mark_sample = ['ro', 'bo', 'go', 'ok']
    mark_centroids = ['Dr', 'Db', 'Dg', 'Dk']
    if k > len(mark_sample):
        print("k is too large")
        return

    fig = plt.figure()
    # plot all samples
    for i in range(EM_Object.sample_count):
        markIndex = EM_Object.result['class'].iloc[i]
        plt.plot(EM_Object.result['sepal length'].iloc[i], EM_Object.result['sepal width'].iloc[i], mark_sample[markIndex])

    # plot centroids
    for i in range(k):
        plt.plot(EM_Object.centroids[i][0, 0], EM_Object.centroids[i][0, 1], mark_centroids[i], markersize=12)

    # set title and labels
    fig.suptitle('Sepal data', fontsize=20)
    plt.xlabel('sepal length in cm', fontsize=18)
    plt.ylabel('sepal width in cm', fontsize=16)
    fig.savefig('Output/cluster_result.png')

    plt.show()





if __name__ == '__main__':
    # Load data and init parameters
    files = ['Input/fisheriris.data','Input/feature.txt']

    # k=3
    k = int(input("Input the number of classes(try k=2,3,4):"))
    EM_data = EM_Object(files,k)
    EM_data.init_par()

    # Clustering data
    cluster(EM_data)

    # Save result
    EM_data.result.to_csv('Output/classification.csv')

    # Plot cluster with sepal's data
    plot(EM_data)