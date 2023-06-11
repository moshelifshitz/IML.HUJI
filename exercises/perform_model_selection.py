from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from IMLearn.metrics import mean_square_error
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    indices = np.random.permutation(X.shape[0])
    train_indices = indices[:n_samples]
    test_indices = indices[:n_samples]

    train_X, train_y = X[train_indices], y[train_indices]
    test_X, test_y = X[test_indices], y[test_indices]
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    min_val_for_ridge = 0
    max_val_for_ridge = 1
    min_val_for_lasso = 0
    max_val_for_lasso = 1
    lamda_parameters_for_ridge = np.linspace(min_val_for_ridge, max_val_for_ridge, n_evaluations)
    lamda_parameters_for_lasso = np.linspace(min_val_for_lasso, max_val_for_lasso, n_evaluations)
    ridge_train_error = np.zeros(n_evaluations)
    ridge_test_error = np.zeros(n_evaluations)
    lasso_train_error = np.zeros(n_evaluations)
    lasso_test_error = np.zeros(n_evaluations)
    for i, lam in enumerate(lamda_parameters_for_ridge):
        ridge_train_error[i], ridge_test_error[i] = cross_validate(RidgeRegression(lam), train_X, train_y,
                                                                   mean_square_error)

    for i, lam in enumerate(lamda_parameters_for_lasso):
        lasso_train_error[i], lasso_test_error[i] = cross_validate(Lasso(lam), train_X, train_y, mean_square_error)

    # Creating subplots with two columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ridge Regression", "Lasso Regression"))

    # Adding the Ridge regression subplot
    fig.add_trace(go.Scatter(x=lamda_parameters_for_ridge,
                             y=ridge_train_error,
                             mode='lines',
                             name='Train Error'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=lamda_parameters_for_ridge,
                             y=ridge_test_error,
                             mode='lines',
                             name='Test Error'),
                  row=1, col=1)

    # Adding the Lasso regression subplot
    fig.add_trace(go.Scatter(x=lamda_parameters_for_lasso,
                             y=lasso_train_error,
                             mode='lines',
                             name='Train Error'),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=lamda_parameters_for_lasso,
                             y=lasso_test_error,
                             mode='lines',
                             name='Test Error'),
                  row=1, col=2)

    # Updating layout
    fig.update_layout(showlegend=False, height=400, width=800, title_text="Regression Error Comparison")

    # Displaying the figure
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_error_for_ridge = np.round(lamda_parameters_for_ridge[np.argmin(ridge_test_error)], 3)
    min_error_for_lasso = np.round(lamda_parameters_for_lasso[np.argmin(lasso_test_error)], 3)
    print("lamda with min error on test for ridge: " + str(min_error_for_ridge))
    print("lamda with min error on test for lasso: " + str(min_error_for_lasso))
    ridge_test_error = RidgeRegression(min_error_for_ridge).fit(train_X, train_y).loss(test_X, test_y)
    lasso_test_y_prediction = Lasso(min_error_for_lasso).fit(train_X, train_y).predict(test_X)
    lasso_test_error = mean_square_error(test_y, lasso_test_y_prediction)
    linear_regression_error = LinearRegression().fit(train_X, train_y).loss(test_X, test_y)
    print("ridge test error with lamda = " + str(min_error_for_ridge) + " is: " + str(ridge_test_error))
    print("lasso test error with lamda = " + str(min_error_for_lasso) + " is: " + str(lasso_test_error))
    print("linear regression  test error is: " + str(linear_regression_error))


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
