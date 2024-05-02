import itertools

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from matplotlib import collections as mc


def plot_margin_example(X, Y, margin=1, C=1.0, ax=None):
    clf = make_pipeline(StandardScaler(), SVC(gamma="auto", C=C))
    clf.fit(X, Y)
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 500),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 500),
    )
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.contour(xx, yy, Z, levels=[0], linewidths=1, colors=["black"])
    ax.scatter(X[Y, 0], X[Y, 1], s=30, c=["r"], edgecolors="k", marker="o", zorder=2)
    ax.scatter(X[~Y, 0], X[~Y, 1], s=30, c=["b"], edgecolors="k", marker="o", zorder=2)

    dist = sp.spatial.distance_matrix(X[Y, :], X[~Y, :], p=2)
    idx1, idx2 = np.where(dist < margin)
    lines = zip(X[Y, :][idx1, :], X[~Y, :][idx2, :])
    ax.add_collection(
        mc.LineCollection(lines, colors="k", linestyle="-", linewidth=0.5, zorder=1)
    )
