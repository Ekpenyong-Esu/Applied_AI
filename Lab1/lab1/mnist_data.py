"""Part 2: load MNIST from the local mnist.npz (same file keras downloads)."""

import numpy as np

from .config import LAB_DIR


def load_mnist(path=LAB_DIR / "mnist.npz"):
    """Returns Xtr (60000, 28, 28), Ltr, X_test (10000, 28, 28), L_test."""
    with np.load(path) as d:
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def flatten(images):
    """(n, 28, 28) -> (n, 784)"""
    return images.reshape(images.shape[0], -1)
