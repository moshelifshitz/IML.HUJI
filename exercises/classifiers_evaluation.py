from math import atan2, pi
from typing import Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from utils import *


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, Y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses_in_each_iteration = []

        # callback function
        def losses_append_loss(fit: Perceptron, _, __):
            losses_in_each_iteration.append(fit.loss(X, Y))

        perceptron_algorithm = Perceptron(callback=losses_append_loss)
        perceptron_algorithm.fit(X, Y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(x=list(range(len(losses_in_each_iteration))), y=losses_in_each_iteration,
                      title="loss as function of fitting iteration for " + n + " dataset").update_layout(
            xaxis_title="num iteration", yaxis_title="error")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        lda = LDA().fit(X, y)
        naive_bayes = GaussianNaiveBayes().fit(X, y)
        naive_bayes_prediction = naive_bayes.predict(X)
        lda_prediction = lda.predict(X)

        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        naive_bayes_accuracy = round(accuracy(y, naive_bayes_prediction) * 100, 2)
        lda_accuracy = round(accuracy(y, lda_prediction) * 100, 2)
        fig3 = make_subplots(cols=2, subplot_titles=(
            "accuracy for Gaussian Naive Bayes: " + str(naive_bayes_accuracy),
            "accuracy for LDA : " + str(lda_accuracy)))

        # Add traces for data-points setting symbols and colors
        fig3.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                  marker=dict(color=naive_bayes_prediction, symbol=class_symbols[y],
                                              colorscale=class_colors(3))), row=1, col=1)

        fig3.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                  marker=dict(color=lda_prediction, symbol=class_symbols[y],
                                              colorscale=class_colors(3))),
                       row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig3.add_trace(go.Scatter(x=naive_bayes.mu_[:, 0], y=naive_bayes.mu_[:, 1], mode="markers",
                                  marker=dict(symbol="x", color="black")), row=1, col=1)

        fig3.add_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                                  marker=dict(symbol="x", color="black")),
                       row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            fig3.add_trace(get_ellipse(naive_bayes.mu_[i], np.diag(naive_bayes.vars_[i])), row=1, col=1)

            fig3.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

        fig3.update_yaxes(scaleanchor="x", scaleratio=1)
        fig3.update_layout(title_text="Comparison of LDA and Gaussian Naive Bayes on " + f)
        fig3.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
