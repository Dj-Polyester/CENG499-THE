import numpy as np


def x_minus_y(x, y):
    '''calculates x-y'''
    y3d = np.stack([np.meshgrid(y_, np.arange(x.shape[0]))[0] for y_ in y])
    return x-y3d


class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):
        '''Negated cosine similarity'''
        return -(y @ x.T)/(np.linalg.norm(y, axis=1)[..., None] @ np.linalg.norm(x, axis=1)[None, ...])

    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        absxy = np.abs(x_minus_y(x, y))
        return (np.sum(absxy**p, axis=2)**(1/p))

    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1=None):
        '''x is the data'''
        if S_minus_1 == None:
            S_minus_1 = np.linalg.inv(np.cov(x.T))
        xy = x_minus_y(x, y)
        s = np.sqrt(xy @ S_minus_1 @ np.transpose(xy, axes=(0, 2, 1)))
        return np.diagonal(s, axis1=1, axis2=2)
