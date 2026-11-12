"""
L1 / L2 distances for the nearest-neighbour classifier.

The bug in the given code (Task 2.1.3): MNIST pixels are uint8, so a - b wraps around
instead of going negative (3 - 5 = 254). Casting to float before subtracting fixes it.
"""

import numpy as np


def l1_distances(train, x):
    return np.sum(np.abs(train.astype(np.float32) - x.astype(np.float32)), axis=1)


def l2_distances(train, x):
    return np.sqrt(np.sum((train.astype(np.float32) - x.astype(np.float32)) ** 2, axis=1))


# the original code, without the cast
def l1_buggy(train, x):
    return np.sum(np.abs(train - x), axis=1)


def l2_buggy(train, x):
    return np.sqrt(np.sum((train - x) ** 2, axis=1))


DISTANCES = {"l1": l1_distances, "l2": l2_distances}
BUGGY_DISTANCES = {"l1": l1_buggy, "l2": l2_buggy}


def uint8_bug_demo(a, b):
    """L1 distance between a and b: buggy code vs fixed code vs computed by hand."""
    by_hand = sum(abs(int(p) - int(q)) for p, q in zip(a.ravel(), b.ravel()))
    return {
        "buggy": int(l1_buggy(a[None, :], b)[0]),
        "fixed": float(l1_distances(a[None, :], b)[0]),
        "by_hand": by_hand,
    }
