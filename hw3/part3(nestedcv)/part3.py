import numpy as np
from DataLoader import DataLoader
from sklearn.base import BaseEstimator
from sklearn.model_selection._validation import check_scoring, _check_multimetric_scoring, indexable, check_cv, is_classifier, _check_fit_params, _fit_and_score, _warn_or_raise_about_fit_failures, _insert_error_scores
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools
from joblib.parallel import Parallel, delayed
from collections import defaultdict
import time


configs = {
    "RandomForestClassifier": {
        "n_estimators": [5, 50],
    },
    "SVC": {
        "kernel": ["linear", "rbf"],
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
    },
}


def name(instance):
    return repr(instance).split("(")[0]


def latexTableBegin(numOfColumns, f=None):
    print(
        f"\\begin{{longtable}}[H]{{{'|c'*numOfColumns+'|'}}}", file=f
    )


def latexTableEnd(caption, label, f=None):
    print(
        f"""    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{longtable}}
""", file=f)


def latexify(*args, slashesAtTheEnd=True, delimiter=" & "):
    return delimiter.join(str(arg) for arg in args) + (" \\\\ \\hline" if slashesAtTheEnd else "")

# https://stackoverflow.com/a/5228294/10713877


def product_dict(**kwargs):
    '''
    Sample output
    {'SVC': {'C': 1, 'degree': None}, 'DecisionTreeClassifier': {'criterion': 'gini', 'splitter': 'best'}}
    {'SVC': {'C': 1, 'degree': None}, 'DecisionTreeClassifier': {'criterion': 'gini', 'splitter': 'random'}}
    '''
    for instances in itertools.product(
        *[itertools.product(*val.values()) for val in kwargs.values()]
    ):
        instancesDics = [
            [dict(zip(val.keys(), instance))] for val, instance in zip(kwargs.values(), instances)
        ]
        yield dict(zip(kwargs.keys(), instancesDics))


class CustomGridSearchCV(GridSearchCV):
    def __init__(self, estimator, param_grid, *, _print=False, fileName=None, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch="2*n_jobs", error_score=np.nan, return_train_score=False):
        super().__init__(estimator, param_grid, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score, return_train_score=return_train_score)
        self.fileName = fileName
        self._print = _print

    @staticmethod
    def refit(cvresults, startsWith="test", newStartsWith="test", returnMean=True):
        estMetricDuos = {
            "_".join(key.split("_")[-2:]) for key in cvresults.keys() if key.startswith(startsWith)
        }
        stacked = np.stack(
            [cvresults[f"{newStartsWith}_{metricName}"] for metricName in estMetricDuos])
        if returnMean:
            return estMetricDuos, stacked.mean(axis=1), stacked.std(axis=1)
        return estMetricDuos, stacked.argmax(axis=1)

    def _select_best_index(self, refit, refit_metric, cvresults):
        cvresults["params"] = np.array(cvresults["params"])
        return refit(cvresults, startsWith="split", newStartsWith="mean_test", returnMean=False)

    @staticmethod
    def print_best_results(fileName, estMetricDuos, bestParams):
        f = None if fileName == None else open(fileName, "a")
        # latexTableBegin(3, f)
        print("\nBest results only showing parameters", file=f)
        print("\\hline", file=f)
        print(latexify("Estimator", "Metric", "Best parameters"), file=f)
        for estMetricDuo, bestParam in zip(estMetricDuos, bestParams):
            estimatorName, metricName = estMetricDuo.split("_")
            params = bestParam[estimatorName]
            print(
                latexify(
                    estimatorName,
                    metricName,
                    *[f"{paramName}={paramValue}" for paramName, paramValue in params.items()]
                ), file=f
            )
        # latexTableEnd("Best results only showing parameters")
        if f != None:
            f.close()

    def print_results(fileName, cvresults, startsWith="split"):
        f = None if fileName == None else open(fileName, "a")
        prevEstimatorName = None
        firstTime = True
        for key in cvresults.keys():
            if key.startswith(startsWith):
                compositeName = key.split("_")
                if len(compositeName) == 4:
                    splitNumber, _, estimatorName, metricName = compositeName
                    splitNumber = int(splitNumber[5:])+1
                    for i, param in enumerate(cvresults["params"]):
                        params = param[estimatorName]
                        if estimatorName != prevEstimatorName:
                            latexTableEnd(
                                f"Results for each split for each metric for {prevEstimatorName}",
                                "label",
                                f,
                            )
                            latexTableBegin(4+len(params.keys()), f)
                            print("\\hline", file=f)
                            print(
                                latexify(
                                    "Split number",
                                    "Estimator name",
                                    "Metric name",
                                    latexify(*params.keys(),
                                             slashesAtTheEnd=False),
                                    "Score"
                                ), file=f
                            )
                        print(
                            latexify(
                                splitNumber,
                                estimatorName,
                                metricName,
                                latexify(*params.values(),
                                         slashesAtTheEnd=False),
                                cvresults[key][i]
                            ), file=f
                        )
                        prevEstimatorName = estimatorName
                elif len(compositeName) == 3:
                    _, estimatorName, metricName = compositeName
                    if firstTime:
                        firstTime = False
                        print("\nBest results only showing scores", file=f)
                        print(
                            latexify(
                                "Estimator name",
                                "Metric name",
                                "Best score for each split",
                            ), file=f)
                    print(
                        latexify(
                            estimatorName,
                            metricName,
                            cvresults[key],
                        ), file=f)
                else:
                    raise ValueError(
                        "The object 'compositeName' is "
                        "not either a list or has "
                        "insufficient number of elements"
                    )
        if f != None:
            f.close()

    def fit(self, X, y=None, *, groups=None, **fit_params):
        # Make bestEstimators None
        self.estimator.bestEstimators = None
        #######################
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in itertools.product(
                        enumerate(candidate_params), enumerate(
                            cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(
                            n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            # self.multimetric_ = isinstance(first_test_score, dict)
            self.multimetric_ = False

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_indices_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            estMetricDuos, self.best_indices_ = self._select_best_index(
                self.refit, refit_metric, results
            )

            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_indices_
                ]
            self.best_params_ = results["params"][self.best_indices_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params_best(
                    estMetricDuos, self.best_params_
                )
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

            if self._print:
                CustomGridSearchCV.print_results(self.fileName, results)
                CustomGridSearchCV.print_best_results(
                    self.fileName, estMetricDuos, self.best_params_)

            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


class EstimatorsWithAPreprocessing(BaseEstimator):
    def __init__(
        self,
        preprocessing,
        estimators,
        metrics,
        times=None,
        bestEstimators=None,
        scorer=None,
    ) -> None:
        super().__init__()

        self.preprocessing = preprocessing
        self.estimators = estimators
        self.metrics = metrics
        self.times = times
        self.bestEstimators = bestEstimators
        self.scorer = scorer

    def setup(self):
        self.estimatorNames = [
            name(estimator) for estimator in self.estimators
        ]
        self.scorer = _MultimetricScorer(
            **{
                metricName: make_scorer(metricFunc) for metricName, metricFunc in self.metrics.items()
            }
        )
        if self.times == None:
            self.times = dict(
                zip(
                    self.estimatorNames,
                    np.ones(len(self.estimatorNames)).astype(np.int32),
                )
            )
        elif isinstance(self.times, (np.ndarray, list)):
            self.times = dict(zip(self.estimatorNames, self.times))
        elif not isinstance(self.times, dict):
            ValueError(
                "Type of the parameter 'times' should be either NoneType, np.ndarray, list or dict"
            )
        if isinstance(self.estimators, (np.ndarray, list)):
            self.estimators = {
                estimatorName: estimator for estimatorName, estimator in zip(self.estimatorNames, self.estimators)
            }
        elif isinstance(self.estimators, dict):
            self.estimators = {
                estimatorName: estimator for estimatorName, estimator in self.estimators.items()
            }
        else:
            ValueError(
                "Type of the parameter 'estimators' should be either np.ndarray, list or dict"
            )

    def setupEstimators(self):
        self.estimators = {
            estimatorName: [clone(estimator) for _ in range(self.times[estimatorName])] for estimatorName, estimator in self.estimators.items()
        }

    def set_params_best(self, estMetricDuos, bestParams):
        self.setup()
        self.bestEstimators = {
            estMetricDuo:
            [clone(
                clone(
                    self.estimators[estMetricDuo.split("_")[0]]
                ).set_params(**bestParam[estMetricDuo.split("_")[0]])
            )]
            for estMetricDuo, bestParam in zip(estMetricDuos, bestParams)
        }
        self.setupEstimators()
        return self

    def set_params(self, **params):
        '''Only set the params of the estimators'''
        self.setup()
        for estimatorName, estimator in self.estimators.items():
            estimator.set_params(**params[estimatorName])
        self.setupEstimators()
        return self

    def scoreEstimatorDic(self, estimatorDic, scorer,  X, y, addScoreName=True):
        return {
            f"{estimatorName}_{scoreName}" if addScoreName else estimatorName: scoreVal
            for estimatorName, estimatorArr in estimatorDic.items()
            for scoreName, scoreVal in self.scoreEstimatorArr(estimatorArr, scorer, X, y)
        }

    def scoreEstimatorArr(self, estimatorArr, scorer,  X, y):
        return zip(self.metrics.keys(), np.stack([
            list(scorer(estimator, X, y).values()) for estimator in estimatorArr
        ]).mean(axis=0))

    def score(self, X_test, y_test):
        '''predict and score test data'''
        X_test = self.fittedPreprocessing.transform(X_test)
        if self.bestEstimators == None:
            return self.scoreEstimatorDic(self.estimators, self.scorer,  X_test, y_test)
        # if self.bestEstimators != None:
        return self.scoreEstimatorDic(self.bestEstimators, self.scorer,  X_test, y_test, False)

    @staticmethod
    def fitEstimatorDic(estimatorDic, X, y, **fit_params):
        for estimator in estimatorDic.values():
            EstimatorsWithAPreprocessing.fitEstimatorArr(
                estimator, X, y, **fit_params
            )

    @staticmethod
    def fitEstimatorArr(estimatorArr, X, y, **fit_params):
        for estimator in estimatorArr:
            estimator.fit(X, y, **fit_params)

    def fit(self, X, y, **fit_params):
        '''fit to the training data'''
        self.fittedPreprocessing = self.preprocessing.fit(X)
        X = self.fittedPreprocessing.transform(X)
        if self.bestEstimators == None:
            EstimatorsWithAPreprocessing.fitEstimatorDic(
                self.estimators, X, y, **fit_params
            )
            return self
        # if self.bestEstimators != None:
        EstimatorsWithAPreprocessing.fitEstimatorDic(
            self.bestEstimators, X, y, **fit_params
        )
        return self


data_path = "data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)


def nestedCV(
    X, y, configsDict, estimators, metrics,
    preprocessing=MinMaxScaler(feature_range=(-1, 1)),
    outerCV=RepeatedStratifiedKFold(n_repeats=5, n_splits=3),
    innerCV=RepeatedStratifiedKFold(n_repeats=5, n_splits=5),
    times=None,
    _print=False,
    fileName=None,
):
    if fileName != None:
        with open(fileName, "w+") as f:
            pass
    # Nested CV with parameter optimization
    innerClf = CustomGridSearchCV(
        estimator=EstimatorsWithAPreprocessing(
            preprocessing, estimators, metrics, times,
        ),
        param_grid=list(product_dict(**configsDict)),
        cv=innerCV,
        error_score="raise",
        refit=CustomGridSearchCV.refit,
        _print=_print,
        fileName=fileName
    )
    cvresults = cross_validate(
        innerClf,
        X=X,
        y=y,
        cv=outerCV,
        error_score="raise",
    )
    if _print:
        CustomGridSearchCV.print_results(
            fileName, cvresults, startsWith="test"
        )
        f = None if fileName == None else open(fileName, "a")
        print("\nOverall results", file=f)
        print("\\hline", file=f)
        print(latexify("Estimator", "Metric", "Score"), file=f)
        for estMetricDuo, meanScore, confidenceScore in zip(*CustomGridSearchCV.refit(cvresults)):
            print(latexify(
                *estMetricDuo.split("_"), f"${meanScore} \\pm {confidenceScore}$"
            ), file=f)
        if f != None:
            f.close()
    return cvresults


if __name__ == "__main__":
    start = time.time()
    print("Starting nested cross validation\n")
    nestedCV(
        X=dataset, y=labels,
        configsDict=configs,
        estimators=[
            RandomForestClassifier(), SVC(), DecisionTreeClassifier()
        ],
        metrics={
            "accuracy": accuracy_score, "f1": f1_score
        },
        times=[1, 5, 1],
        _print=True,
        fileName="results.txt",
    )
    print(
        f"\nCompleted nested cross validation in {time.time()-start} seconds")
