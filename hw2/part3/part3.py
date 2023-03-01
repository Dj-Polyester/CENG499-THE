import pickle
from sklearn.cluster import AgglomerativeClustering
import itertools
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
import time
plt.rcParams['figure.figsize'] = (15, 10)


class Silhouette:
    def __init__(self, model: AgglomerativeClustering) -> None:
        self.model = model
        self.n_samples = model.labels_.shape[0]
        self.distMatrix = None

    def computeDistance(self, leaf1, leaf2, nodeIndex=-1):
        nodeIndexRefined = nodeIndex if nodeIndex < self.n_samples else nodeIndex-self.n_samples
        left, right = self.model.children_[nodeIndexRefined]
        l1 = self.inTree(leaf1, left)
        l2 = self.inTree(leaf2, left)
        r1 = not l1
        r2 = not l2
        if (l1 and l2):
            return self.computeDistance(leaf1, leaf2, left)
        elif (r1 and r2):
            return self.computeDistance(leaf1, leaf2, right)
        elif (l1 and r2) or (l2 and r1):
            return self.model.distances_[nodeIndexRefined]

    def inTree(self, leaf, nodeIndex):
        if nodeIndex < self.n_samples:  # leaf node
            return leaf == nodeIndex
        left, right = self.model.children_[nodeIndex-self.n_samples]
        return self.inTree(leaf, left) or self.inTree(leaf, right)

    def computeDistanceMatrix(self):
        distMatrix = np.zeros((self.n_samples, self.n_samples))
        for j in range(self.n_samples):
            for i in range(j+1, self.n_samples):
                distMatrix[j, i] = self.computeDistance(j, i)
        self.distMatrix = distMatrix+distMatrix.T
        return self.distMatrix

    def score(self):
        self.computeDistanceMatrix()
        return silhouette_score(self.distMatrix, self.model.labels_)

# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html


def plot_dendrogram(model, **kwargs):
    '''Creates linkage matrix and then plots the dendrogram, Use plt.plot or plt.savefig after this function'''

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Index of the point")
    plt.ylabel("Distance")
    dendrogram(linkage_matrix, **kwargs)


configs = {
    "affinity": ["euclidean", "cosine"],
    "linkage": ["single", "complete"],
    "n_clusters": [2, 3, 4, 5],
    "compute_distances": [True],
}

STRLITERAL = "Iteration & Distance metric & Linkage & Number of clusters & Silhouette Score & Time"

# https://stackoverflow.com/a/5228294/10713877


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


dataset = pickle.load(open("data/part3_dataset.data", "rb"))

if __name__ == "__main__":
    start = time.time()
    length = len(list(product_dict(**configs)))
    print(STRLITERAL)
    for i, kwargs in enumerate(product_dict(**configs)):
        model = AgglomerativeClustering(**kwargs)
        model = model.fit(dataset)
        silhouette = Silhouette(model)
        silhouetteScore = silhouette.score()
        kwargsrepr = [str(v) for v in kwargs.values()]
        kwargsrepr.pop()
        kwargsLatex = " & ".join([f"${kwarg}$" for kwarg in kwargsrepr])
        print(
            f"({i+1}/{length}) & {kwargsLatex} & ${silhouetteScore}$ & ${time.time()-start}$")

        if kwargs["n_clusters"] == 2:
            fileName = "_".join(kwargsrepr)
            plot_dendrogram(model)
            plt.savefig(f"{fileName}_dendrogram.png")
            plt.clf()
