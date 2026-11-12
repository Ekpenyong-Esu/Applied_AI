"""
Part 4: experiments on 10 of the 121 UCI datasets from Fernandez-Delgado et al. (2014),
"Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?".

Every run: stratified 80/20 split with its own seed, preprocessing inside a Pipeline
(so it is only fitted on the training part), GridSearchCV with 5 folds on the training
part for the supervised models, one score on the test part. Averaged over N_SEEDS.

The clustering models never see labels while fitting; afterwards each cluster gets the
majority label of the training points in it, and that mapping is used on the test set.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .config import SEED

# OpenML name -> name in the paper, and the paper's best accuracy on it (TODO: fill in)
DATASETS = {
    "iris":          {"delgado_name": "iris",                  "baseline": None},
    "wine":          {"delgado_name": "wine",                  "baseline": None},
    "glass":         {"delgado_name": "glass",                 "baseline": None},
    "ionosphere":    {"delgado_name": "ionosphere",            "baseline": None},
    "sonar":         {"delgado_name": "connectionist-bench-sonar", "baseline": None},
    "balance-scale": {"delgado_name": "balance-scale",         "baseline": None},
    "breast-w":      {"delgado_name": "breast-cancer-wisc",    "baseline": None},
    "diabetes":      {"delgado_name": "pima",                  "baseline": None},
    "vehicle":       {"delgado_name": "statlog-vehicle",       "baseline": None},
    "ecoli":         {"delgado_name": "ecoli",                 "baseline": None},
}

N_SEEDS = 5
SEEDS = [SEED + i for i in range(N_SEEDS)]


def load_dataset(name):
    bunch = fetch_openml(name, version=1, as_frame=True)
    X = bunch.data
    y = LabelEncoder().fit_transform(bunch.target.astype(str))
    info = {
        "dataset": name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
    }
    return X, y, info


def describe_datasets():
    rows = []
    for name in DATASETS:
        _, _, info = load_dataset(name)
        info["delgado_name"] = DATASETS[name]["delgado_name"]
        rows.append(info)
    return pd.DataFrame(rows)


def make_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


class ClusterClassifier(BaseEstimator, ClassifierMixin):
    """Fit a clusterer without labels, then label each cluster by majority vote."""

    def __init__(self, clusterer):
        self.clusterer = clusterer

    def fit(self, X, y):
        self.clusterer_ = clone(self.clusterer).fit(X)
        clusters = self.clusterer_.predict(X)
        self.classes_ = np.unique(y)
        self.cluster_to_label_ = {c: np.bincount(y[clusters == c]).argmax() for c in np.unique(clusters)}
        # used if a test point lands in a cluster that got no training points
        self.fallback_ = np.bincount(y).argmax()
        return self

    def predict(self, X):
        clusters = self.clusterer_.predict(X)
        return np.array([self.cluster_to_label_.get(c, self.fallback_) for c in clusters])


def build_models(n_classes, seed=SEED):
    """{name: (estimator, param grid or None)}"""
    return {
        "SVM-RBF (supervised)": (
            SVC(kernel="rbf"),
            {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1]},
        ),
        "RandomForest (supervised)": (
            RandomForestClassifier(random_state=seed),
            {"model__n_estimators": [100, 300], "model__max_features": ["sqrt", None]},
        ),
        "KMeans (unsupervised)": (
            ClusterClassifier(KMeans(n_clusters=n_classes, n_init=10, random_state=seed)),
            None,
        ),
        "GaussianMixture (unsupervised)": (
            ClusterClassifier(GaussianMixture(n_components=n_classes, random_state=seed)),
            None,
        ),
    }


def run_one_experiment(X, y, model_name, estimator, param_grid, seed, cv_folds=5):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe = Pipeline([("prep", make_preprocessor(X)), ("model", estimator)])
    if param_grid:
        pipe = GridSearchCV(pipe, param_grid, cv=cv_folds, n_jobs=-1)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    return {
        "model": model_name,
        "seed": seed,
        "accuracy": accuracy_score(y_te, pred),
        "f1_macro": f1_score(y_te, pred, average="macro"),
        "best_params": getattr(pipe, "best_params_", None),
    }


def run_all(seeds=SEEDS):
    """One row per (dataset, model, seed)."""
    rows = []
    for name in DATASETS:
        X, y, info = load_dataset(name)
        for model_name, (estimator, grid) in build_models(info["n_classes"]).items():
            accs = []
            for seed in seeds:
                row = run_one_experiment(X, y, model_name, estimator, grid, seed)
                row["dataset"] = name
                rows.append(row)
                accs.append(row["accuracy"])
            print(f"{name:<14} {model_name:<32} mean acc = {np.mean(accs):.3f}")
    return pd.DataFrame(rows)


def summarise(runs):
    """Mean/std over seeds per (dataset, model), next to the paper's number."""
    table = (
        runs.groupby(["dataset", "model"])
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
             f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"))
        .reset_index()
    )
    table["paper_baseline"] = table["dataset"].map(lambda d: DATASETS[d]["baseline"])
    table["diff_vs_paper"] = table["accuracy_mean"] - table["paper_baseline"]
    return table


def best_model_per_dataset(table):
    idx = table.groupby("dataset")["accuracy_mean"].idxmax()
    return table.loc[idx, ["dataset", "model", "accuracy_mean", "f1_mean"]].reset_index(drop=True)
