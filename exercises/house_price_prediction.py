import math
from datetime import datetime

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
TRAIN = None


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X = X.replace(['NA', 'N/A', 'nan', None, np.nan], None)
    if y is not None:
        y = y.astype(float)
        X['price'] = y
        X = X[X['price'] >= 1]
        X = X.dropna(subset='price')
    X = X.drop(['id', 'sqft_living15', 'sqft_lot15', 'long', 'lat', 'date'], axis=1)
    # change None vals to col mean
    # Use original values for non-numeric columns
    for col in X.columns:
        X[col] = pd.to_numeric(X[col])
    for col in X.columns:
        col_mean = X[col][X[col] >= 0].mean()
        X = X.replace(['NA', 'N/A', 'nan', None, np.nan], None)
        X[col] = X[col].apply(lambda x: col_mean if x is None else x)
        X[col] = X[col].apply(lambda x: col_mean if x < 0 else x)
    # change yr_renovated col to true false val and change it's nae to renovated

    # calculate the current year
    current_year = datetime.now().year
    # add a new column with 1 if year is within last 20 years, 0 otherwise
    X['renovated_in_last_20_years'] = X['yr_renovated'].apply(lambda x: 1 if current_year - x <= 20 else 0)
    # drop irrelevant vals
    X = X.drop(['yr_renovated'], axis=1)
    # change to dummies some elements
    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])

    if y is not None:
        # map of minimum val
        min_val_map = {'bedrooms': 1, 'bathrooms': 0, 'sqft_living': 1, 'sqft_lot': 1,
                       'floors': 1, 'waterfront': 0, 'view': 0, 'condition': 1, 'grade': 0, 'sqft_above': 0,
                       'yr_built': 0}
        for col in min_val_map.keys():
            X = X[X[col] >= min_val_map[col]]
        # map of maximum val
        max_val_map = {'bedrooms': 10, 'bathrooms': 5, 'sqft_lot': 1000000,
                       'floors': 5, 'waterfront': 1, 'view': 4, 'condition': 5, 'grade': 14}
        for col in max_val_map.keys():
            X = X[X[col] <= max_val_map[col]]
        global TRAIN
        TRAIN = X.drop('price', axis=1)
        return X.drop('price', axis=1), X['price']

    if y is None:
        return X.reindex(columns=TRAIN.columns, fill_value=0)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False))]
    for col in X.columns:
        pearson_correlation = np.cov(X[col], y)[0, 1] / (np.std(X[col]) * np.std(y))
        fig1 = px.scatter(x=X[col], y=y,
                          title=f"The pearson correlation between {col} and responding vector: " + str(
                              np.round(pearson_correlation, 3))).update_layout(
            xaxis_title="feature " + col,yaxis_title="actual price")
        fig1.show()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    X = df.drop(['price'], axis=1)
    y = df['price']
    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_Y = split_train_test(X, y)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_Y = test_Y.replace(['NA', 'N/A', 'nan', None, np.nan], None)
    test_Y = test_Y.dropna()
    test_X = test_X.loc[test_Y.index]
    test_X = preprocess_data(test_X)
    res = np.empty((91, 10))
    for i in range(10, 101):
        for j in range(10):
            p_present_sample_x = train_X.sample(frac=(i / 100.0))
            p_present_sample_y = train_y.loc[p_present_sample_x.index]
            linear_regression = LinearRegression()
            linear_regression.fit(p_present_sample_x, p_present_sample_y)
            res[i - 10, j] = linear_regression.loss(test_X, test_Y)
    average = res.mean(axis=1)
    var = res.std(axis=1)

    fig = go.Figure([go.Scatter(x=list(range(10,101)), y=average - 2 * var, fill=None, mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=list(range(10,101)), y=average + 2 * var, fill='tonexty', mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=list(range(10,101)), y=average, mode="markers+lines", marker=dict(color="black"))],
                    layout=go.Layout(title="mse of p percent of the samples",
                                     xaxis=dict(title="percent"),
                                     yaxis=dict(title="MSE"),
                                     showlegend=False))
    fig.show()
