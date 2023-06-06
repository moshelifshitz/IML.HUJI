from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # find all classe
        self.classes_, count_classes = np.unique(y, return_counts=True)

        # calculate pi
        self.pi_ = count_classes / len(y)

        # calculate mu
        mean = []
        for class_ in self.classes_:
            mask = (y == class_)
            relevant_rows = X[mask]
            relevant_rows_mean = np.mean(relevant_rows, axis=0)
            mean.append(relevant_rows_mean)
        self.mu_ = np.array(mean)

        # calculate vars
        # self.vars_ = np.zeros((len(self.classes_), X.shape[1]), dtype=np.float)
        # for i, class_ in enumerate(self.classes_):
        #     self.vars_[i] = np.var(X[y == class_], ddof=1, axis=0)

        var = []
        for class_ in self.classes_:
            mask = (y == class_)
            relevant_rows = X[mask]
            relevant_rows_var = np.var(relevant_rows, axis=0, ddof=1)
            var.append(relevant_rows_var)
        self.vars_ = np.array(var)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        exp_argument = ((X[:, np.newaxis, :] - self.mu_) ** 2) / (-2 * self.vars_)
        factor = np.sqrt(2 * self.vars_ * np.pi)
        exp_expression = np.exp(exp_argument) / factor
        return np.prod(exp_expression, axis=2) * self.pi_

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
        from ...metrics import misclassification_error
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)
