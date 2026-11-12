"""Accuracy, confusion matrix and macro F1 in plain numpy (Part 2 doesn't use sklearn)."""

import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred):
    """Rows = true class, columns = predicted class."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    index = {lab: i for i, lab in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        C[index[t], index[p]] += 1
    return C


def macro_f1(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    tp = np.diag(C).astype(float)
    precision = tp / np.maximum(C.sum(axis=0), 1)
    recall = tp / np.maximum(C.sum(axis=1), 1)
    with np.errstate(invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return float(f1.mean())
