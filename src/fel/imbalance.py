"""Study 2: how accuracy misleads when one class dominates (simulated data).

The data is generated. It is not the lab's student data, and the numbers it produces
are labelled "simulated". The class shares follow the shape of the distribution table in
the research report (one class near 92%, the rest small), not its exact values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fel.metrics import classification_metrics, summarise_folds
from fel.pipeline import SklearnAdapter
from fel.splits import grouped_kfold

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEFAULT_SHARES = {  # majority class = surprise
    "surprise": 0.92, "happy": 0.026, "angry": 0.024, "sad": 0.023,
    "neutral": 0.004, "fear": 0.002, "disgust": 0.001,
}


def make_user_dataset(
    n_users: int = 120,
    rows_per_user: int = 100,
    n_features: int = 20,
    class_sep: float = 0.3,
    user_scale: float = 0.8,
    noise: float = 1.0,
    shares: dict[str, float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """X, y (indices into CLASS_NAMES), groups (user id), labels. Simulated."""
    shares = shares or DEFAULT_SHARES
    p = np.array([shares[c] for c in CLASS_NAMES], dtype=float)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, class_sep, size=(len(CLASS_NAMES), n_features))
    X, y, g = [], [], []
    for u in range(n_users):
        offset = rng.normal(0.0, user_scale, size=n_features)
        yy = rng.choice(len(CLASS_NAMES), size=rows_per_user, p=p)
        X.append(means[yy] + offset + rng.normal(0.0, noise, size=(rows_per_user, n_features)))
        y.append(yy)
        g.append(np.full(rows_per_user, u))
    return np.vstack(X), np.concatenate(y), np.concatenate(g), list(CLASS_NAMES)


def oversample(X: np.ndarray, y: np.ndarray, cap: int = 2000, seed: int = 0):
    """Duplicate minority rows (training rows only) up to `cap` per class. Never the majority."""
    rng = np.random.default_rng(seed)
    parts_X, parts_y = [X], [y]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if 0 < len(idx) < cap:
            extra = rng.choice(idx, size=cap - len(idx), replace=True)
            parts_X.append(X[extra])
            parts_y.append(y[extra])
    return np.vstack(parts_X), np.concatenate(parts_y)


def _make(name: str, balanced: bool, n_classes: int, seed: int):
    cw = "balanced" if balanced else None
    if name == "knn9":
        est = KNeighborsClassifier(n_neighbors=9)
    elif name == "logreg":
        est = LogisticRegression(max_iter=500, class_weight=cw, random_state=seed)
    elif name == "linear_svc":
        est = LinearSVC(class_weight=cw, random_state=seed, max_iter=5000)
    else:
        raise ValueError(name)
    return SklearnAdapter(est, n_classes, use_proba=(name != "linear_svc"))


STRATEGIES = ("baseline", "class_weight", "oversample")
MODELS = ("knn9", "logreg", "linear_svc")


def run_imbalance_study(seed: int = 0, n_splits: int = 5, **data_kw) -> dict[str, Any]:
    """Every model with every strategy, evaluated with user-grouped k-fold. Simulated."""
    X, y, g, labels = make_user_dataset(seed=seed, **data_kw)
    idx_labels = list(range(len(labels)))
    rows = []
    for model in MODELS:
        for strat in STRATEGIES:
            if strat == "class_weight" and model == "knn9":
                continue  # k-NN has no class weights
            folds = []
            for k, (tr, te) in enumerate(grouped_kfold(y, g, n_splits, seed=seed)):
                sc = StandardScaler().fit(X[tr])
                Xtr, ytr = sc.transform(X[tr]), y[tr]
                if strat == "oversample":
                    Xtr, ytr = oversample(Xtr, ytr, seed=seed + k)
                m = _make(model, strat == "class_weight", len(labels), seed).fit(Xtr, ytr)
                pred = m.predict_proba(sc.transform(X[te])).argmax(axis=1)
                folds.append(classification_metrics(y[te], pred, idx_labels))
            s = summarise_folds(folds)
            rows.append({
                "name": f"{model} / {strat}",
                "accuracy": s["accuracy_mean"], "accuracy_std": s["accuracy_std"],
                "macro_f1": s["macro_f1_mean"], "macro_f1_std": s["macro_f1_std"],
            })
    majority = float(np.max(np.bincount(y)) / len(y))
    return {
        "table": rows,
        "majority_class_share": majority,
        "always_predict_majority_accuracy": majority,
        "n_rows": int(len(y)),
        "n_users": int(len(np.unique(g))),
        "split": f"user-grouped {n_splits}-fold",
    }
