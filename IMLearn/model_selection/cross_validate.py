from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    errors_for_train = np.zeros(cv)
    errors_for_test = np.zeros(cv)
    indices = np.arange(X.shape[0])
    folds = np.array_split(indices, cv)
    for i in range(cv):
        samples_indices = np.concatenate(folds[:i] + folds[i + 1:])
        test_indices = folds[i]
        k_samples_X = X[samples_indices]
        k_samples_y = y[samples_indices]
        k_tests_X = X[test_indices]
        k_tests_y = y[test_indices]
        estimator.fit(k_samples_X, k_samples_y)
        y_pred_train = estimator.predict(k_samples_X)
        y_pred_test = estimator.predict(k_tests_X)
        errors_for_train[i] = scoring(k_samples_y, y_pred_train)
        errors_for_test[i] = scoring(k_tests_y, y_pred_test)
    return errors_for_train.mean(), errors_for_test.mean()

