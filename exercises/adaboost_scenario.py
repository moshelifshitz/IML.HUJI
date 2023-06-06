import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    num_iteration = list(range(1, 251))
    train_losses_per_num_iterations = np.empty((249,))
    test_losses_per_num_iterations = np.empty((249,))
    for i in range(249):
        train_losses_per_num_iterations[i] = adaboost.partial_loss(train_X, train_y, i + 1)
        test_losses_per_num_iterations[i] = adaboost.partial_loss(test_X, test_y, i + 1)

    fig1 = go.Figure(
        data=[go.Scatter(x=num_iteration, y=train_losses_per_num_iterations, name="Train Loss per num classifiers"),
              go.Scatter(x=num_iteration, y=test_losses_per_num_iterations, name="Test Loss per num classifiers")])
    fig1.update_layout(xaxis_title="num classifiers", yaxis_title="loss",
                       title="loss on train and test samples for 1-250 classifiers")
    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig2 = make_subplots(rows=1, cols=4, subplot_titles=[str(num) + " Classifiers" for num in T])
    for i, num_classifiers in enumerate(T):
        fig2.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, num_classifiers),
                                          lims[0], lims[1]),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y,
                                                symbol=np.where(test_y == 1, "circle", "x")))],
                        rows=1, cols=i + 1)
    fig2.show()

    # Question 3: Decision surface of best performing ensemble

    best_performence = np.argmin(test_losses_per_num_iterations) + 1
    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                              marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x"))))

    fig3.add_trace(decision_surface(lambda X: adaboost.partial_predict(X, best_performence), lims[0], lims[1]))

    fig3.update_layout(title="Accuracy: " + str(1 - round(test_losses_per_num_iterations[best_performence - 1], 2)) +
                             " | Size of best performing: " + str(best_performence),
                       xaxis=dict(visible=False), yaxis=dict(visible=False))

    fig3.show()

    # Question 4: Decision surface with weighted samples

    D = 20 * adaboost.D_ / adaboost.D_.max()
    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                              marker=dict(size=D, color=train_y, symbol=np.where(train_y == 1, "circle", "x"))))

    fig4.add_trace(decision_surface(adaboost.predict, lims[0], lims[1]))

    fig4.update_layout(title="size of weights at the end of the algorithm",
                       xaxis=dict(visible=False), yaxis=dict(visible=False))

    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
