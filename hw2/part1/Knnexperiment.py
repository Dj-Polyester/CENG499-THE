import pickle
from Knn import KNN
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import itertools
import time
import types
import sys
import os
from Distance import Distance
# from importlib import import_module

# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
# _dist = import_module("Distance")
# Distance = getattr(_dist, "Distance")


configs = {
    "similarityFunction": [Distance.calculateCosineDistance, Distance.calculateMinkowskiDistance, Distance.calculateMahalanobisDistance],
    "similarityFunctionParameters": [None, 1, 2],
    "K": [1, 5, 9],
}

STRLITERAL = "Iteration & Distance function & Auxillary parameters & K & Test accuracy & Time"

# https://stackoverflow.com/a/5228294/10713877


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        tmpdic = dict(zip(keys, instance))
        c1 = tmpdic["similarityFunction"] == Distance.calculateMinkowskiDistance
        c2 = tmpdic["similarityFunctionParameters"] == None
        if (c1 and not c2) or (not c1 and c2):
            yield dict(zip(keys, instance))


dataset, labels = pickle.load(open("data/part1_dataset.data", "rb"))

if __name__ == "__main__":
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    start = time.time()
    length = len(list(product_dict(**configs)))
    splitLength = rskf.get_n_splits()
    print(STRLITERAL)
    for i, kwargs in enumerate(product_dict(**configs)):
        testaccs = np.zeros(splitLength)
        split = 0
        for trainIndex, testIndex in rskf.split(dataset, labels):
            X_train, X_test = dataset[trainIndex], dataset[testIndex]
            y_train, y_test = labels[trainIndex], labels[testIndex]
            knn = KNN(X_train, y_train, **kwargs)
            knn.predict(X_test)
            testaccs[split] = knn.eval(y_test)
            split += 1

        kwargsrepr = [v.__name__ if type(
            v) == types.FunctionType else str(v) for v in kwargs.values()]
        kwargsLatex = " & ".join([f"${kwarg}$" for kwarg in kwargsrepr])

        testaccsMargin = 1.96 * (testaccs.std()/np.sqrt(splitLength))
        print(f"({i+1}/{length}) & {kwargsLatex} & ${testaccs.mean()} \pm {testaccsMargin}$ & ${time.time()-start}$")
