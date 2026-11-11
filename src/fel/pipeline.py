"""Cross-validation runners: the corrected protocol and the legacy protocol, side by side.

A "model" here is any object with fit(X, y, X_val=None, y_val=None) and predict_proba(X).
`PreprocessedModel` adds Mutual Information selection and scaling that are fitted on the
fitting rows only, so nothing from validation or test rows reaches them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from fel.features import MutualInfoSelector
from fel.metrics import bootstrap_ci_by_group, classification_metrics, summarise_folds
from fel.splits import group_overlap, grouped_kfold, inner_validation_split, random_frame_split


class Model(Protocol):
    def fit(self, X, y, X_val=None, y_val=None): ...
    def predict_proba(self, X) -> np.ndarray: ...


class SklearnAdapter:
    """Wrap a scikit-learn classifier so predict_proba always has one column per class."""

    def __init__(self, estimator, n_classes: int, use_proba: bool = True):
        self.estimator, self.n_classes, self.use_proba = estimator, n_classes, use_proba

    def fit(self, X, y, X_val=None, y_val=None):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        out = np.zeros((len(X), self.n_classes))
        classes = np.asarray(self.estimator.classes_, dtype=int)
        if self.use_proba and hasattr(self.estimator, "predict_proba"):
            out[:, classes] = self.estimator.predict_proba(X)
        else:  # one-hot of the hard prediction (for models without probabilities)
            pred = np.asarray(self.estimator.predict(X), dtype=int)
            out[np.arange(len(X)), pred] = 1.0
        return out


class PreprocessedModel:
    """Selection and scaling fitted on the fitting rows only, then the model."""

    def __init__(self, model: Model, keep_fraction: float = 1.0, scale: bool = True,
                 seed: int = 0):
        self.model, self.keep_fraction, self.scale, self.seed = model, keep_fraction, scale, seed

    def _tx(self, X):
        X = self.selector_.transform(X)
        return self.scaler_.transform(X) if self.scaler_ is not None else X

    def fit(self, X, y, X_val=None, y_val=None):
        self.selector_ = MutualInfoSelector(self.keep_fraction, self.seed).fit(X, y)
        Xs = self.selector_.transform(X)
        self.scaler_ = StandardScaler().fit(Xs) if self.scale else None
        Xf = self.scaler_.transform(Xs) if self.scaler_ is not None else Xs
        Xv = self._tx(X_val) if X_val is not None else None
        self.model.fit(Xf, y, Xv, y_val)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self._tx(X))


ModelFactory = Callable[[float], Model]  # argument: Mutual Information keep fraction


def encode_labels(y: Sequence[Any]) -> tuple[np.ndarray, list[str]]:
    labels = sorted({str(v) for v in y})
    idx = {lab: i for i, lab in enumerate(labels)}
    return np.array([idx[str(v)] for v in y]), labels


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    make_model: ModelFactory,
    keep_fraction: float = 1.0,
    n_splits: int = 5,
    grouped: bool = True,
    seed: int = 0,
    n_boot: int = 500,
) -> dict[str, Any]:
    """k-fold cross-validation. grouped=True splits by subject, False splits random rows."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if grouped:
        folds = list(grouped_kfold(y, groups, n_splits=n_splits, seed=seed))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(np.zeros(len(y)), y))
    idx_labels = list(range(len(labels)))
    fold_metrics, true_all, pred_all, grp_all, overlaps = [], [], [], [], []
    for k, (tr, te) in enumerate(folds):
        overlaps.append(group_overlap(tr, te, groups))
        if grouped:
            fit_idx, val_idx = inner_validation_split(tr, groups, 0.2, seed + k)
        else:
            fit_idx, val_idx = tr[: int(0.8 * len(tr))], tr[int(0.8 * len(tr)):]
        model = make_model(keep_fraction)
        model.fit(X[fit_idx], y[fit_idx], X[val_idx], y[val_idx])
        proba = model.predict_proba(X[te])
        pred = proba.argmax(axis=1)
        fold_metrics.append(classification_metrics(y[te], pred, idx_labels, proba))
        true_all.append(y[te])
        pred_all.append(pred)
        grp_all.append(groups[te])
    yt, yp, gg = np.concatenate(true_all), np.concatenate(pred_all), np.concatenate(grp_all)
    pooled = classification_metrics(yt, yp, idx_labels)
    pooled["labels"] = list(labels)
    lo, hi = bootstrap_ci_by_group(yt, yp, gg, idx_labels, "macro_f1", n_boot, seed=seed)
    return {
        "protocol": "grouped k-fold" if grouped else "random k-fold (rows)",
        "n_splits": n_splits,
        "keep_fraction": keep_fraction,
        "mean_groups_shared_between_train_and_test": float(np.mean(overlaps)),
        "folds": summarise_folds(fold_metrics),
        "pooled": pooled,
        "macro_f1_ci95_over_subjects": [lo, hi],
    }


def run_legacy_protocol(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    make_model: ModelFactory,
    keep_fraction: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """The thesis protocol: select on ALL rows, random 70/15/15 row split, test set read once.

    Kept only to measure how much it inflates results on the same features.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    selector = MutualInfoSelector(keep_fraction, seed).fit(X, y)  # sees test labels: the leak
    Xs = selector.transform(X)
    tr, va, te = random_frame_split(len(y), seed=seed)
    model = make_model(1.0)
    model.fit(Xs[tr], y[tr], Xs[va], y[va])
    proba = model.predict_proba(Xs[te])
    idx_labels = list(range(len(labels)))
    m = classification_metrics(y[te], proba.argmax(axis=1), idx_labels, proba)
    m["labels"] = list(labels)
    return {
        "protocol": "legacy (selection on all rows, random 70/15/15 row split)",
        "keep_fraction": keep_fraction,
        "groups_shared_between_train_and_test": group_overlap(tr, te, groups),
        "n_groups_in_test": int(len(set(groups[te]))),
        "test": m,
    }
