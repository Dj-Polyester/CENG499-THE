import numpy as np
import math
from Distance import Distance
from KMeans import KMeans


class KMeansPlusPlus:
    KMeans

    def __init__(self, dataset, K=2, initializer=KMeans.KMeansPlusPlus, similarityFunction=Distance.calculateMinkowskiDistance, similarityFunctionParameters=2):
        super.__init__(self, dataset, K, initializer,
                       similarityFunction, similarityFunctionParameters)
