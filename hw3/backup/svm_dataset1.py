import pickle
import numpy as np
import itertools
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time

FILEEXT = ".png"
configs = {
    "C": [1e-7, int(1e6)],
    "kernel": ["linear", "poly", "rbf"],
    "degree": [None, 1, 2, 3],
}

STRLITERAL = "Iteration & C & Kernel function & Degree & Time"

# https://stackoverflow.com/a/5228294/10713877


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        tmpdic = dict(zip(keys, instance))
        c1 = tmpdic["kernel"] == "poly"
        c2 = tmpdic["degree"] == None
        if (c1 and not c2) or (not c1 and c2):
            if c2:
                tmpdic["degree"] = 1
            yield tmpdic


def plotContour2d(svc, X, Y, minx=-1.1, maxx=1.1, miny=-1.1, maxy=1.1, samplesx=500, samplesy=500):
    xx, yy = np.meshgrid(np.linspace(minx, maxx, samplesx),
                         np.linspace(miny, maxy, samplesy))
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=1.5, linestyles="dashed")
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
    plt.axis([minx, maxx, miny, maxy])


dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))

if __name__ == "__main__":
    start = time.time()
    print(STRLITERAL)
    length = len(list(product_dict(**configs)))
    for i, kwargs in enumerate(product_dict(**configs)):
        fileName = "_".join([str(kwarg) for kwarg in kwargs.values()])+FILEEXT
        latexRepr = " & ".join([f"${kwarg}$" for kwarg in kwargs.values()])
        svc = SVC(**kwargs)
        svc.fit(dataset, labels)
        plotContour2d(svc, dataset, labels)
        plt.savefig(fileName)
        plt.clf()
        print(f"({i+1}/{length}) & {latexRepr} & ${time.time()-start}$")
