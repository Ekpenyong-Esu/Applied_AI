"""Task 2.2: pick k for k-NN with k-fold cross-validation on the training set only."""

import numpy as np

from .config import SEED
from .metrics import accuracy
from .nearest_neighbour import NearestNeighbor


def k_fold_indices(n_samples, n_folds=3, seed=SEED):
    perm = np.random.default_rng(seed).permutation(n_samples)
    return np.array_split(perm, n_folds)


def cross_validate_k(X, y, k_values, n_folds=3, distance="l2", seed=SEED):
    """Returns {k: (mean accuracy, [accuracy per fold])}."""
    folds = k_fold_indices(X.shape[0], n_folds, seed)
    results = {}
    for k in k_values:
        fold_scores = []
        for v in range(n_folds):
            va = folds[v]
            tr = np.concatenate([f for i, f in enumerate(folds) if i != v])
            model = NearestNeighbor(k=k, distance=distance).train(X[tr], y[tr])
            fold_scores.append(accuracy(y[va], model.predict(X[va])))
        results[k] = (float(np.mean(fold_scores)), fold_scores)
    return results


def best_k(results):
    # on a tie the smaller k wins
    return min(results, key=lambda k: (-results[k][0], k))
