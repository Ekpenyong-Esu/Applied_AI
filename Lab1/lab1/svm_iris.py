"""
Part 3: SVM on Iris, following the "SVM clearly explained" TDS blog
(80/20 split, svm.SVC(kernel=..., C=1, gamma='auto')).

SVC is one-vs-one internally, so to compare the two strategies the same SVC is wrapped
in OneVsOneClassifier and OneVsRestClassifier (3 binary SVMs each for 3 classes).
"""

from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from .config import SEED

KERNELS = ("linear", "poly", "rbf")
STRATEGIES = {"one-vs-one": OneVsOneClassifier, "one-vs-rest": OneVsRestClassifier}
# 0-based column indices for "features 2 vs 3" and "features 3 vs 4"
FEATURE_PAIRS = {"features 2 vs 3": (1, 2), "features 3 vs 4": (2, 3)}


def load_iris():
    iris = datasets.load_iris()
    return iris.data, iris.target, list(iris.feature_names), list(iris.target_names)


def split_80_20(X, y, seed=SEED):
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def make_svm(strategy, kernel):
    return STRATEGIES[strategy](svm.SVC(kernel=kernel, C=1.0, gamma="auto"))


def train_all(X_train, y_train):
    """Returns {(strategy, kernel): fitted model}."""
    return {(s, k): make_svm(s, k).fit(X_train, y_train) for s in STRATEGIES for k in KERNELS}


def evaluate_all(models, X_test, y_test):
    matrices, scores = {}, []
    for (strategy, kernel), model in models.items():
        pred = model.predict(X_test)
        matrices[(strategy, kernel)] = confusion_matrix(y_test, pred)
        scores.append({
            "strategy": strategy,
            "kernel": kernel,
            "accuracy": accuracy_score(y_test, pred),
            "f1_macro": f1_score(y_test, pred, average="macro"),
        })
    return matrices, scores


def best_by(scores, metric):
    return max(scores, key=lambda row: row[metric])


def support_vectors_one_vs_rest(ovr_model, X_train, y_train, class_names):
    """
    OneVsRestClassifier keeps one binary SVC per class in estimators_. Its support_
    indices are split into vectors from class c and vectors from the other classes.
    """
    out = {}
    for c, est in enumerate(ovr_model.estimators_):
        idx = est.support_
        is_c = y_train[idx] == c
        out[class_names[c]] = {
            "n_total": len(idx),
            "n_from_class": int(is_c.sum()),
            "n_from_rest": int((~is_c).sum()),
            "vectors_from_class": X_train[idx[is_c]],
            "vectors_from_rest": X_train[idx[~is_c]],
        }
    return out


def train_on_feature_pair(X_train, y_train, pair, kernel="rbf", strategy="one-vs-rest"):
    """Train on just two columns so the decision boundary can be drawn in 2-D."""
    X2 = X_train[:, list(pair)]
    return make_svm(strategy, kernel).fit(X2, y_train), X2
