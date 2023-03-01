import numpy as np
from importlib import import_module
import sys
import os
from types import NoneType
from Distance import Distance

# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
# _dist = import_module("Distance")
# Distance = getattr(_dist, "Distance")

EPSILON = .0001


class KMeans:
    def __init__(self, dataset, K=2, initializer=None, similarityFunction=Distance.calculateMinkowskiDistance, similarityFunctionParameters=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        if initializer == None:
            initializer = KMeans.KMeansMinusMinus
        self.K = K
        self.dataset = dataset
        self.similarityFunction = similarityFunction
        self.similarityFunctionParameters = similarityFunctionParameters
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        # self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        # self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class
        self.clusterCenters = initializer(self)
        self.clusterLabels = None

    @staticmethod
    def KMeansMinusMinus(self):
        """Default KMeans initialization, bad D:"""
        return self.dataset[np.random.choice(len(self.dataset), self.K, replace=False)]

    @staticmethod
    def KMeansPlusPlus(self):
        """KMeans++ initialization, yeyyyy :D"""
        # first center is chosen uniformly
        lenX = len(self.dataset)
        clusterCenters = self.dataset[np.random.choice(lenX)]
        for i in range(1, self.K):
            # calculate new weights
            args = (clusterCenters, self.dataset) if self.similarityFunctionParameters == None else (
                clusterCenters, self.dataset, self.similarityFunctionParameters)
            dist = self.similarityFunction(*args)
            clusterDists2 = np.min(dist, axis=1)**2
            p = clusterDists2 / clusterDists2.sum()
            # choose next center
            newClusterCenter = self.dataset[np.random.choice(lenX, p=p)]
            clusterCenters = np.concatenate(
                (clusterCenters[None, ...] if i == 1 else clusterCenters, newClusterCenter[None, ...]))
        return clusterCenters

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        loss = 0
        for i in range(self.K):
            args = (self.clusterCenters[i], self.dataset[self.clusterLabels == i]) if self.similarityFunctionParameters == None else (
                self.clusterCenters[i], self.dataset[self.clusterLabels == i], self.similarityFunctionParameters)
            loss += self.similarityFunction(*args).sum()
        return loss

    def run(self):
        """Kmeans algorithm implementation"""
        args = (self.clusterCenters, self.dataset) if self.similarityFunctionParameters == None else (
            self.clusterCenters, self.dataset, self.similarityFunctionParameters)
        # modify until cluster means dont change
        clusterCentersPrev = None
        while (type(clusterCentersPrev) == NoneType) or ((clusterCentersPrev - self.clusterCenters) >= EPSILON).any():
            clusterCentersPrev = self.clusterCenters.copy()
            # step 1, label data points as closest clusters
            dist = self.similarityFunction(*args)
            self.clusterLabels = np.argmin(dist, axis=1)
            # step 2, update cluster centers as the mean of clusters
            self.clusterCenters = np.stack(
                [self.dataset[self.clusterLabels == i].mean(axis=0) for i in range(self.K)])

        return self.clusterCenters, self.clusterLabels, self.calculateLoss()
