from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet

NUM_SAMPLES = 1000


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        division_factor = (len(X) - 1)
        if self.biased_:
            division_factor += 1
        self.mu_ = X.mean()
        samples_minus_expectation = X - self.mu_
        samples_minus_expectation_square = samples_minus_expectation * samples_minus_expectation
        self.var_ = samples_minus_expectation_square.sum() / division_factor
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        temp = 1 / (2 * self.var_) * ((X - self.mu_) * (X - self.mu_))
        factor = (1 / math.sqrt(2 * np.pi * self.var_))
        ret = factor * np.exp(-temp)
        return ret

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # the log_likelihood is -m/2 * log(2pi*sigma) - (1/(2*sigma))*(sum(xi - mu)^2)
        first_elem = (X.shape[0] / 2) * math.log(2 * math.pi * sigma)
        X_minus_mu_square = (X - mu) * (X - mu)
        second_elem = (1 / (2 * sigma)) * X_minus_mu_square.sum()
        return - first_elem - second_elem


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        centered = X - self.mu_
        self.cov_ = (1 / (X.shape[0] - 1)) * (centered.T @ centered)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        random_vectors_dimension = X.shape[1]
        cov_det = np.linalg.det(self.cov_)
        factor = 1 / np.sqrt(math.pow(2 * math.pi, random_vectors_dimension) * cov_det)
        X_minus_mu = X - self.mu_
        inner_expression = (1 / 2) * np.diag(
            X_minus_mu @ np.linalg.inv(self.cov_) @ X_minus_mu.transpose())
        pdf_exp_part = np.exp(-inner_expression)
        return factor * pdf_exp_part

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        vectors_dimension = X.shape[1]
        cov_det = np.linalg.det(cov)
        num_samples = X.shape[0]
        X_minus_mu = X - mu
        first_part = (num_samples / 2) * np.log(math.pow(2 * math.pi, vectors_dimension))
        second_part = (num_samples / 2) * math.log(cov_det)
        third_part = (1 / 2) * np.trace(
            X_minus_mu @ np.linalg.inv(cov) @ X_minus_mu.transpose())
        return - first_part - second_part - third_part
