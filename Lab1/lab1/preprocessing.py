"""
Task 1.1 step 3, following https://hadrienj.github.io/posts/Preprocessing-for-deep-learning/
Function names are kept from the tutorial. X is always (n_samples, n_features).
"""

import numpy as np


def calculateCovariance(X):
    """Same as np.cov(X, rowvar=False, bias=True), done by hand."""
    Xc = X - np.mean(X, axis=0)
    return Xc.T.dot(Xc) / X.shape[0]


def simulate_uncorrelated(n=300, seed=1234):
    rng = np.random.RandomState(seed)
    return np.array([rng.normal(2, 1, n), rng.normal(1, 1, n)]).T


def simulate_correlated(n=300, seed=1234):
    rng = np.random.RandomState(seed)
    b1 = rng.normal(3, 1, n)
    b2 = b1 + rng.normal(7, 1, n) / 2.0
    return np.array([b1, b2]).T


def center(X):
    return X - np.mean(X, axis=0)


def standardize(X):
    return center(X) / np.std(X, axis=0)


def decorrelate(X):
    """Rotate onto the eigenvectors of the covariance matrix -> diagonal covariance."""
    Xc = center(X)
    _, eigVecs = np.linalg.eigh(calculateCovariance(Xc))
    return Xc.dot(eigVecs)


def whiten(X):
    """Decorrelate, then divide by sqrt(eigenvalue) -> identity covariance."""
    Xc = center(X)
    eigVals, eigVecs = np.linalg.eigh(calculateCovariance(Xc))
    return Xc.dot(eigVecs) / np.sqrt(eigVals + 1e-5)


def zca_whitening(X_norm, epsilon=0.1):
    """X_zca = U diag(1/sqrt(S + eps)) U^T X, with U, S from the SVD of the pixel covariance."""
    cov = np.cov(X_norm, rowvar=False)
    U, S, _ = np.linalg.svd(cov)
    return U.dot(np.diag(1.0 / np.sqrt(S + epsilon))).dot(U.T).dot(X_norm.T).T


def preprocess_my_images(X_images):
    """Flatten, scale to 0..1, subtract the per-pixel mean, ZCA. Returns every step."""
    flat = X_images.reshape(X_images.shape[0], -1)
    unit = flat / 255.0
    centred = unit - unit.mean(axis=0)
    zca = zca_whitening(centred)
    return {
        "flat": flat,
        "unit": unit,
        "centred": centred,
        "zca": zca,
        # min-max back to 0..1 so imshow can draw it
        "zca_display": (zca - zca.min()) / (zca.max() - zca.min()),
    }
