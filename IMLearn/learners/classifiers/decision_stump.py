from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # errors matrix which hold the errors for all features with sign = 1 in the first row
        # and sign = -1 in the second row
        errors = np.empty((2, X.shape[1]))
        # thresholds matrix which hold the thresholds for all features with sign = 1 in the first row
        # and sign = -1 in the second row
        thresholds = np.empty((2, X.shape[1]))
        for f_ind in range(X.shape[1]):
            for s_ind, sign in enumerate([1, -1]):
                feature = X[:, f_ind]
                thresholds[s_ind, f_ind], errors[s_ind, f_ind] = self._find_threshold(feature, y, sign)

        min_index = np.unravel_index(np.argmin(errors), errors.shape)
        self.j_ = min_index[1]
        self.threshold_ = thresholds[min_index]
        if min_index[0] == 0:
            self.sign_ = 1
        else:
            self.sign_ = -1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_pred = np.empty((X.shape[0],))
        b_sign_indices = np.where(X[:, self.j_] >= self.threshold_)
        b_minus_sign_indices = np.where(X[:, self.j_] < self.threshold_)
        y_pred[b_sign_indices] = self.sign_
        y_pred[b_minus_sign_indices] = -self.sign_
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # sort labels
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # initialize array for losses
        thresholds_loss_array = np.sum(np.abs(labels[np.sign(labels) == sign]))
        thresholds_loss_array = np.append(thresholds_loss_array,
                                          thresholds_loss_array - np.cumsum(sorted_labels * sign))
        min_index = np.argmin(thresholds_loss_array)
        if min_index == 0:
            threshold = -np.inf
        elif min_index < len(values) - 1:
            threshold = sorted_values[min_index]
        else:
            threshold = np.inf
        return threshold, thresholds_loss_array[min_index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
