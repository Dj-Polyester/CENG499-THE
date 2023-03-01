import pickle
import numpy as np
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator


class EstimatorWithAPreprocessing(BaseEstimator):
    def __init__(self, preprocessing, estimator) -> None:
        super().__init__()
        self.preprocessing = preprocessing
        self.estimator = estimator

    def set_params(self, **params):
        '''Only set the params of the last element'''
        self.estimator.set_params(**params)
        return self

    def fit(self, X, y, **fit_params):
        '''fit to the training data'''
        self.fittedPreprocessing = self.preprocessing.fit(X)
        X = self.fittedPreprocessing.transform(X)

        return self.estimator.fit(X, y, **fit_params)

    def predict(self, X):
        '''predict test data'''
        X = self.fittedPreprocessing.transform(X)
        return self.estimator.predict(X)


FILEEXT = ".png"
configs = {
    "C": [1, 10],
    "kernel": ["linear", "poly", "rbf"],
    "degree": [None, 1, 2, 3],
}


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
            tmpdiclone = {}
            for k, v in tmpdic.items():
                tmpdiclone[k] = [v]
            yield tmpdiclone


dataset, labels = pickle.load(open("data/part2_dataset2.data", "rb"))


def latexify(*args, slashesAtTheEnd=True, delimiter=" & "):
    return delimiter.join(str(arg) for arg in args) + (" \\\\ \\hline" if slashesAtTheEnd else "")


def preprocessDataInGridSearch(estimator, preprocessing, cvSplitter):
    gridSearchCV = GridSearchCV(
        EstimatorWithAPreprocessing(preprocessing, estimator),
        cv=cvSplitter,
        scoring="accuracy",
        param_grid=list(product_dict(**configs)),
        refit=False,  # Not necessary for this task
    )
    gridSearchCV.fit(dataset, labels)
    return gridSearchCV


def print_results(cvresults, bestparams, n_repeats, fileName=None, startsWith="split"):
    f = None if fileName == None else open(fileName, "w+")
    firstTime = True
    for key in cvresults.keys():
        if key.startswith(startsWith):
            splitNumber, _,  __ = key.split("_")
            splitNumber = int(splitNumber[5:])+1
            for i, param in enumerate(cvresults["params"]):
                if firstTime:
                    firstTime = False
                    print("\\hline", file=f)
                    print(
                        latexify(
                            "Split number",
                            latexify(*param.keys(),
                                     slashesAtTheEnd=False),
                            "Score"
                        ), file=f
                    )
                print(
                    latexify(
                        splitNumber,
                        latexify(*param.values(),
                                 slashesAtTheEnd=False),
                        cvresults[key][i]
                    ), file=f
                )

    acc = cvresults["mean_test_score"]
    # confidence interval with level 95 percent
    accMargin = 1.96*(
        cvresults["std_test_score"] /
        np.sqrt(n_repeats)
    )
    paramCombinations = np.vstack(
        (
            np.array([
                cvresults[f"param_{k}"] for k in configs.keys()
            ]),
            acc,
            accMargin,
        )
    )
    print("\n\\hline", file=f)
    print(latexify("C", "Kernel function", "Degree",
          "Mean accuracy"), file=f)
    for tup in zip(*paramCombinations):
        print(latexify(
            latexify(*tup[:-2], slashesAtTheEnd=False),
            f"${tup[-2]} \pm {tup[-1]}$"
        ), file=f)
    print(f"Best: {tuple(bestparams.values())}", file=f)

    if f != None:
        f.close()


def GridSearchWithRepeatedStratifiedKFold(
    procedure=preprocessDataInGridSearch,
    estimator=SVC(),
    preprocessing=StandardScaler(),
    fileName=None,
    _print=False,
    **kwargs
):
    if "n_splits" not in kwargs:
        kwargs["n_splits"] = 10
    if "n_repeats" not in kwargs:
        kwargs["n_repeats"] = 5
    cvSplitter = RepeatedStratifiedKFold(**kwargs)

    gridSearchCV = procedure(estimator, preprocessing, cvSplitter)
    if _print:
        print_results(
            gridSearchCV.cv_results_,
            gridSearchCV.best_params_,
            kwargs["n_repeats"],
            fileName,
        )


if __name__ == "__main__":
    GridSearchWithRepeatedStratifiedKFold(fileName="results.txt", _print=True)
