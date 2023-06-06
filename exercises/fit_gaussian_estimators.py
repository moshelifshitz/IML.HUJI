import math

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import sys

sys.path.append("../")
from utils import *
from scipy.stats import norm

pio.templates.default = "simple_white"
NUM_SAMPLES = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    univariate_gaussian = UnivariateGaussian()
    samples = np.random.normal(10, 1, NUM_SAMPLES)
    univariate_gaussian.fit(samples)
    expectation, var = univariate_gaussian.mu_, univariate_gaussian.var_
    print("(" + str(round(expectation, 3)) + ", " + str(round(var, 3)) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    distances = list(range(int(NUM_SAMPLES / 10)))
    key = list(range(NUM_SAMPLES // 10))

    # distances = np.empty((int(NUM_SAMPLES / 10),))
    # key = np.empty((int(NUM_SAMPLES / 10),))
    univariate_gaussian_for_samples = UnivariateGaussian()
    for num_samples in range(10, NUM_SAMPLES + 1, 10):
        m_samples = samples[0:num_samples]
        estimated_expectation = univariate_gaussian_for_samples.fit(m_samples).mu_
        distances[(num_samples // 10) - 1] = abs(estimated_expectation - expectation)
        key[(num_samples // 10) - 1] = num_samples

    # plot
    fig1 = go.Figure([go.Scatter(x=key, y=distances, mode='markers', marker=dict(color="black"), showlegend=False)],
                     layout=go.Layout(
                         title_text=r"$\text{(1) The expectation's distances for m samples from 1000 samples}$",
                         height=300, width=500, xaxis_title="Number of samples",
                         yaxis_title="Distance"))

    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = univariate_gaussian.pdf(samples)
    fig2 = go.Figure([go.Scatter(x=samples, y=pdf, mode='markers', marker=dict(color="black"), showlegend=False)],
                     layout=go.Layout(title_text=r"$\text{(1) PDF distribution}$",
                                      height=300,width=500, xaxis_title="number of samples",
                                      yaxis_title="PDF"))

    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov, NUM_SAMPLES)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(samples)
    estimated_mu, estimated_cov = multivariate_gaussian.mu_, multivariate_gaussian.cov_
    print(np.round(estimated_mu, 3), "\n", np.round(estimated_cov, 3))

    # Question 5 - Likelihood evaluation
    F = np.linspace(-10, 10, 200)
    max_f1, max_f3 = None, None
    max_val = - math.inf
    log_likelihood_matrix = np.zeros((200, 200))
    for i, f1 in enumerate(F):
        for j, f3 in enumerate(F):
            log_likelihood_matrix[i, j] = multivariate_gaussian.log_likelihood(np.array([f1, 0, f3, 0]),
                                                                               multivariate_gaussian.cov_, samples)
            if log_likelihood_matrix[i, j] > max_val:
                max_f1 = f1
                max_f3 = f3
                max_val = log_likelihood_matrix[i, j]
    fig3 = go.Figure(go.Heatmap(x=F, y=F, z=log_likelihood_matrix),
                     layout=go.Layout(title_text=r"$\text{the graph for f1, f3 in range [-10,10]}$",
                                      height=500, width=500, xaxis_title="f3",
                                      yaxis_title="f1"))
    fig3.show()

    # Question 6 - Maximum likelihood
    print(np.round(max_f1, 3), np.round(max_f3, 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
