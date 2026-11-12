"""
Task 1.2: the collaborator's vecs.npy.

It's a dict of dicts: data['3_pos'][digit] -> list of 1024-dim embeddings.
"""

import numpy as np

from .config import SEED, VECS_FILE


def load_vecs(path=VECS_FILE):
    # np.save pickled the dict into a 0-d object array, .item() gets it back
    return np.load(path, allow_pickle=True).item()


def describe_vecs(data):
    lines = []
    total = 0
    for pos_key, digits in data.items():
        n_pos = sum(len(v) for v in digits.values())
        total += n_pos
        dims = {np.asarray(v[0]).shape for v in digits.values() if len(v)}
        lines.append(f"{pos_key}: digits {sorted(digits)} -> {n_pos} samples, vector shape {dims}")
    lines.append(f"Total samples: {total}")
    return "\n".join(lines)


def arrays_for_position(data, position):
    """Embeddings (m, 1024) and labels (m,) for one position, in the order of the file."""
    digits = data[f"{position}_pos"]
    X = np.concatenate([np.asarray(digits[d], dtype=np.float32) for d in digits])
    y = np.concatenate([np.full(len(digits[d]), int(d)) for d in digits])
    return X, y


def shuffle_together(X, y, seed=SEED):
    """Same permutation for X and y so the pairs stay matched."""
    perm = np.random.default_rng(seed).permutation(X.shape[0])
    return X[perm], y[perm], perm
