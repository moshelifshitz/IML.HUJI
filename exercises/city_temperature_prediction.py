import plotly.graph_objects as go

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    X = pd.read_csv(filename, parse_dates=["Date"])
    X = X[X["Temp"] > -5]
    X["Year"] = X["Year"].astype(str)
    X = X.dropna().drop_duplicates()
    X['DayOfYear'] = X['Date'].dt.dayofyear
    return X


if __name__ == '__main__':
    np.random.seed(0)
    X = load_data("../datasets/City_Temperature.csv")
    # Question 1 - Load and preprocessing of city temperature dataset

    # Question 2 - Exploring data for specific country
    X_israel = X[X['Country'] == "Israel"]
    fig1 = px.scatter(X_israel, x='DayOfYear', y='Temp', color='Year')
    fig1.update_layout(title="temp avg in Israel as function of day of year",
                       xaxis_title="day of year", yaxis_title="temp avg")
    fig1.show()

    fig2 = px.bar(X_israel.groupby(["Month"], as_index=False).agg(std=("Temp", "std")),
                  title="monthly temperature deviation", x="Month", y="std")
    fig2.show()

    # Question 3 - Exploring differences between countries
    fig3 = px.line(X.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std")),
                   x="Month", y="mean", error_y="std", color="Country") \
        .update_layout(title="monthly temperatures avg",
                       xaxis_title="month",
                       yaxis_title="avg temperature")
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(X_israel.DayOfYear, X_israel.Temp)
    loss_val_for_k = np.zeros(shape=(10,))
    for i in range(1, 11):
        polynomial_fitting = PolynomialFitting(k=i).fit(train_X.to_numpy(), train_y.to_numpy())
        loss_val_for_k[i - 1] = np.round(polynomial_fitting.loss(test_X.to_numpy(), test_y.to_numpy()), 2)

    loss_val_for_k_df = pd.DataFrame(dict(k=list(range(1, 11)), loss=loss_val_for_k))
    fig4 = px.bar(loss_val_for_k_df, x="k", y="loss", text="loss",
                  title="errors for polynomial in degree 1 <= k <= 10 in IL")
    fig4.show()
    print(loss_val_for_k_df)

    # Question 5 - Evaluating fitted model on different countries
    polynomial_fitting = PolynomialFitting(k=5).fit(X_israel.DayOfYear.to_numpy(), X_israel.Temp.to_numpy())
    fig5 = px.bar(pd.DataFrame(
        [{"country": country,
          "loss": round(polynomial_fitting.loss(X[X.Country == country].DayOfYear, X[X.Country == country].Temp), 2)}
         for country in ["Jordan", "South Africa", "The Netherlands"]]),
        x="country", y="loss", text="loss", color="country",
        title="loss of the rest of the countries.png")
    fig5.show()
