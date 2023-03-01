import numpy as np


class KNN:
    def __init__(self, dataset, dataLabel, similarityFunction, similarityFunctionParameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics, PASSED IN DICTIONARY
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.datasetLabel = dataLabel
        self.similarityFunction = similarityFunction
        self.similarityFunctionParameters = similarityFunctionParameters

        self.preds = None

    def predict(self, instances):
        args = (self.dataset, instances) if self.similarityFunctionParameters == None else (
            self.dataset, instances, self.similarityFunctionParameters)

        dist = self.similarityFunction(*args)
        closestKIndices = np.argsort(dist, axis=1)[:, :self.K]
        closestLabels = self.datasetLabel[closestKIndices]
        occs = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=np.argmax(
                self.datasetLabel)+1), 1, closestLabels
        )
        self.preds = np.argmax(occs, axis=1)
        return self.preds

    def eval(self, labels):
        return 100*(self.preds == labels).mean()
