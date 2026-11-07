"""Metrics for imbalanced multi-class problems.

Accuracy alone can hide classes that are never predicted. Every report from this
repo carries macro F1 and per-class recall next to accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def expected_calibration_error(
    y_true_idx: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Expected calibration error with equal-width confidence bins.

    y_true_idx: integer class indices. y_prob: (n, n_classes) probabilities.
    """
    y_true_idx = np.asarray(y_true_idx)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2 or len(y_prob) != len(y_true_idx):
        raise ValueError("y_prob must have shape (n_samples, n_classes)")
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if lo == 0.0:
            mask = mask | (conf == 0.0)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any],
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Accuracy, macro F1, per-class precision/recall/F1/support, confusion matrix.

    `labels` fixes the class order. Classes with no true samples are reported with
    support 0. `never_predicted` lists classes that appear in y_true but are never
    predicted.
    """
    labels = list(labels)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("no samples")
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        str(lab): {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for lab, p, r, f, s in zip(labels, prec, rec, f1, sup, strict=True)
    }
    predicted = set(y_pred.tolist())
    never_predicted = [str(lab) for lab, s in zip(labels, sup, strict=True)
                       if s > 0 and lab not in predicted]
    out: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted",
                                      zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "never_predicted": never_predicted,
        "labels": [str(lab) for lab in labels],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if y_prob is not None:
        idx = {lab: i for i, lab in enumerate(labels)}
        y_idx = np.array([idx[v] for v in y_true.tolist()])
        out["ece"] = expected_calibration_error(y_idx, y_prob)
    return out


def bootstrap_ci_by_group(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    groups: Sequence[Any],
    labels: Sequence[Any],
    metric: str = "macro_f1",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval that resamples whole groups (subjects).

    Resampling subjects, not rows, because rows from one subject are correlated.
    """
    if metric not in ("macro_f1", "accuracy"):
        raise ValueError("metric must be 'macro_f1' or 'accuracy'")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("need at least 2 groups for a group bootstrap")
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    labels = list(labels)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        if metric == "accuracy":
            stats[b] = accuracy_score(y_true[idx], y_pred[idx])
        else:
            stats[b] = f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro",
                                zero_division=0)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarise_folds(fold_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of accuracy and macro F1 across folds."""
    if not fold_metrics:
        raise ValueError("no folds")
    acc = np.array([m["accuracy"] for m in fold_metrics])
    f1 = np.array([m["macro_f1"] for m in fold_metrics])
    return {
        "n_folds": len(fold_metrics),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
    }
