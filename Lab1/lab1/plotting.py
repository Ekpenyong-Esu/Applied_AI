"""All the figures. Each function returns the Figure so it can be saved for the report."""

import numpy as np
from matplotlib import pyplot as plt

from .config import FIGURES_DIR


def save(fig, name):
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def PlotSample(X, index, y=None, names=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure
    ax.imshow(X[index])
    title = f"sample {index}"
    if y is not None:
        label = int(y[index])
        title += f" - label {label}" + (f" ({names[label]})" if names else "")
    ax.set_title(title)
    ax.axis("off")
    return fig


def plot_samples_grid(X, y, names, indices, n_cols=5):
    indices = list(indices)
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, idx in zip(np.ravel(axes), indices):
        PlotSample(X, idx, y, names, ax=ax)
    fig.tight_layout()
    return fig


def plotDataAndCov(data, title=""):
    """Like the tutorial's plotDataAndCov: covariance heatmap next to a scatter of the data."""
    ACov = np.cov(data, rowvar=False, bias=True)
    print(f"Covariance matrix {title}:\n", ACov)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    im = ax1.imshow(ACov, cmap="viridis")
    for (i, j), v in np.ndenumerate(ACov):
        ax1.text(j, i, f"{v:.2f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax1)
    ax1.set_title("covariance matrix")
    ax2.scatter(data[:, 0], data[:, 1], s=8)
    ax2.set_title(f"data {title}")
    ax2.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def plot_before_after_whitening(X_unit, X_zca_display, index, shape):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6, 3))
    a.imshow(X_unit[index].reshape(shape))
    a.set_title("original (scaled 0..1)")
    b.imshow(X_zca_display[index].reshape(shape))
    b.set_title("ZCA whitened")
    for ax in (a, b):
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_mnist_pair(Xtr, Ltr, i=0, j=1):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.imshow(Xtr[i]); ax0.set_title(f"Label is {Ltr[i]}")
    ax1.imshow(Xtr[j]); ax1.set_title(f"Label is {Ltr[j]}")
    return fig


def plot_cv_results(results, title="3-fold cross-validation"):
    """Mean validation accuracy vs k, grey dots are the single folds."""
    ks = sorted(results)
    means = [results[k][0] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k in ks:
        ax.scatter([k] * len(results[k][1]), results[k][1], color="grey", s=12)
    ax.plot(ks, means, marker="o", label="mean over folds")
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("validation accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_confusion_matrix(C, class_names, title="", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
    else:
        fig = ax.figure
    ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(C):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > C.max() / 2 else "black")
    return fig


def plot_confusion_grid(matrices, class_names):
    """matrices = {(strategy, kernel): C}, strategies as rows and kernels as columns."""
    strategies = sorted({s for s, _ in matrices})
    kernels = sorted({k for _, k in matrices})
    fig, axes = plt.subplots(len(strategies), len(kernels),
                             figsize=(3.4 * len(kernels), 3.2 * len(strategies)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(strategies):
        for c, k in enumerate(kernels):
            plot_confusion_matrix(matrices[(s, k)], class_names, f"{s} / {k}", ax=axes[r, c])
    fig.tight_layout()
    return fig


def plot_decision_boundary(model, X2, y, feature_names, title="", h=0.02, ax=None):
    """Classify every point of a fine meshgrid and colour the regions (as in the SVM blog)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    return fig


def plot_results_bar(table, metric="accuracy_mean", title=""):
    pivot = table.pivot(index="dataset", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(1.2 * len(pivot) + 3, 4))
    pivot.plot.bar(ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
