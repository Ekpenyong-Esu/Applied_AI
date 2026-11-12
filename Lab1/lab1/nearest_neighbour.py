"""Part 2: the lecturer's 1-NN extended to k-NN with majority voting."""

import numpy as np

from .distances import BUGGY_DISTANCES, DISTANCES


class NearestNeighbor:
    def __init__(self, k=1, distance="l1", fix_dtype=True):
        # fix_dtype=False runs the original uint8 code so we can show the bug
        self.k = k
        self.distance = distance
        self.fix_dtype = fix_dtype
        self.dist_fn = (DISTANCES if fix_dtype else BUGGY_DISTANCES)[distance]

    def train(self, X, y):
        self.Tr_set = X
        self.Ltr_set = y
        return self

    def predict_one(self, x):
        distances = self.dist_fn(self.Tr_set, x)
        nearest = np.argpartition(distances, self.k - 1)[: self.k]
        # ties go to the smallest label
        return np.argmax(np.bincount(self.Ltr_set[nearest]))

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def __repr__(self):
        return f"NearestNeighbor(k={self.k}, distance={self.distance!r}, fix_dtype={self.fix_dtype})"
