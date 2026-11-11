"""Study runners shared by the scripts and the notebooks. Every function returns a plain dict
that results.write_result can save. Tables are lists of rows with accuracy and macro F1."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from fel import landmarks as L
from fel.imbalance import run_imbalance_study
from fel.models import HAS_TORCH, TorchFERClassifier
from fel.pipeline import (
    PreprocessedModel,
    SklearnAdapter,
    encode_labels,
    run_cv,
    run_legacy_protocol,
)
from fel.synthetic import make_sequence_dataset


def _row(name: str, accs: list[float], f1s: list[float]) -> dict[str, Any]:
    return {"name": name, "accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "n_repeats": len(accs)}


def make_factory(model: str, n_classes: int, seed: int = 0, balance: bool = False):
    """Return make_model(keep_fraction).

    Models: majority, logreg, knn1, mlp, fernet (needs PyTorch).
    """
    def make(keep: float):
        if model == "majority":
            core = SklearnAdapter(DummyClassifier(strategy="most_frequent"), n_classes)
        elif model == "logreg":
            core = SklearnAdapter(
                LogisticRegression(max_iter=300, class_weight="balanced" if balance else None,
                                   random_state=seed), n_classes)
        elif model == "knn1":
            core = SklearnAdapter(KNeighborsClassifier(1), n_classes)
        elif model == "mlp":
            core = SklearnAdapter(MLPClassifier((128,), max_iter=300, early_stopping=True,
                                                random_state=seed), n_classes)
        elif model == "fernet":
            if not HAS_TORCH:
                raise ImportError("fernet needs PyTorch: pip install -e '.[torch]'")
            core = TorchFERClassifier(n_classes, balance=balance, seed=seed)
        else:
            raise ValueError(f"unknown model {model!r}")
        return PreprocessedModel(core, keep, seed=seed)
    return make


def leakage_demo(seeds=range(5), n_splits: int = 5) -> dict[str, Any]:
    """Simulated: the same classifiers under three protocols on sequence-structured data."""
    rows, per = [], {}
    for model in ("logreg", "knn1", "mlp"):
        for s in seeds:
            X, y, g = make_sequence_dataset(seed=s)
            labels = ["a", "b", "c", "d"]
            mk = make_factory(model, 4, seed=s)
            runs = {
                "legacy (random 70/15/15 rows, selection on all rows)":
                    partial(run_legacy_protocol, X, y, g, labels, mk, 0.5, seed=42),
                "random 5-fold (rows)":
                    partial(run_cv, X, y, g, labels, mk, 0.5, n_splits, False, s, 20),
                "subject-wise 5-fold":
                    partial(run_cv, X, y, g, labels, mk, 0.5, n_splits, True, s, 20),
            }
            for proto, fn in runs.items():
                r = fn()
                m = r["test"] if "test" in r else r["pooled"]
                per.setdefault((model, proto), []).append((m["accuracy"], m["macro_f1"]))
    for (model, proto), vals in per.items():
        a, f = zip(*vals, strict=True)
        rows.append(_row(f"{model} | {proto}", list(a), list(f)))
    return {"table": rows, "seeds": list(seeds), "chance_accuracy": 0.25,
            "note": "simulated sequence data: 30 subjects, 4 classes, 4 near-identical frames "
                    "per sequence"}


def selection_leak_demo(seeds=range(5), n: int = 200, p: int = 300, keep: float = 0.04):
    """Simulated: features are pure noise. Compares selecting on all rows with selecting
    inside each training fold. Reports what is measured, whatever the size of the gap."""
    rows = {"selected on all rows, then random 70/15/15 split": [],
            "selected inside each training fold, random 4-fold": []}
    labels = list("abcd")
    mk = make_factory("logreg", 4)
    for s in seeds:
        rng = np.random.default_rng(s)
        X, y, g = rng.normal(size=(n, p)), rng.integers(0, 4, n), np.arange(n)
        t = run_legacy_protocol(X, y, g, labels, mk, keep, seed=s)["test"]
        rows["selected on all rows, then random 70/15/15 split"].append(
            (t["accuracy"], t["macro_f1"]))
        c = run_cv(X, y, g, labels, mk, keep, 4, False, s, 10)["pooled"]
        rows["selected inside each training fold, random 4-fold"].append(
            (c["accuracy"], c["macro_f1"]))
    table = []
    for name, vals in rows.items():
        a, f = zip(*vals, strict=True)
        table.append(_row(name, list(a), list(f)))
    return {"table": table, "chance_accuracy": 0.25, "n_rows": n, "n_noise_features": p,
            "keep_fraction": keep}


def imbalance_demo(seed: int = 0) -> dict[str, Any]:
    return run_imbalance_study(seed=seed)


# ---------------- Study 1 on extracted landmarks ----------------

def load_features_npz(path: str) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def build_feature_sets(landmarks: np.ndarray, blendshapes: np.ndarray) -> dict[str, np.ndarray]:
    d3 = L.distance_features(landmarks, use_z=True)
    d2 = L.distance_features(landmarks, use_z=False)
    pts = L.point_features(landmarks, use_z=True)
    return {
        "distances 2D": d2,
        "distances 3D": d3,
        "distances 3D + blendshapes": np.hstack([d3, blendshapes]),
        "distances 3D + points + blendshapes": np.hstack([d3, pts, blendshapes]),
    }


def run_study1(
    data: dict[str, np.ndarray],
    feature_sets: list[str] | None = None,
    models: list[str] | None = None,
    keep: float = 0.75,
    n_splits: int = 5,
    seed: int = 0,
    n_boot: int = 300,
) -> dict[str, Any]:
    """Grouped CV (the corrected protocol), random-row CV and the legacy protocol on the same
    features. Returns tables only: no paths, no per-subject numbers."""
    y, labels = encode_labels(data["labels"])
    groups = data["groups"]
    fsets = build_feature_sets(data["landmarks"], data["blendshapes"])
    names = feature_sets or list(fsets)
    models = models or ["majority", "logreg", "mlp"]
    table, detail = [], {}
    for fs in names:
        X = fsets[fs]
        for model in models:
            mk = make_factory(model, len(labels), seed)
            k = 1.0 if model == "majority" else keep
            grouped = run_cv(X, y, groups, labels, mk, k, n_splits, True, seed, n_boot)
            f = grouped["folds"]
            table.append({"name": f"{fs} | {model} | subject-wise {n_splits}-fold",
                          "accuracy": f["accuracy_mean"], "accuracy_std": f["accuracy_std"],
                          "macro_f1": f["macro_f1_mean"], "macro_f1_std": f["macro_f1_std"],
                          "macro_f1_ci95_over_subjects": grouped["macro_f1_ci95_over_subjects"]})
            if model != "majority":
                rnd = run_cv(X, y, groups, labels, mk, k, n_splits, False, seed, 20)
                lg = run_legacy_protocol(X, y, groups, labels, mk, k, seed=42)
                f = rnd["folds"]
                table.append({"name": f"{fs} | {model} | random {n_splits}-fold (rows)",
                              "accuracy": f["accuracy_mean"], "accuracy_std": f["accuracy_std"],
                              "macro_f1": f["macro_f1_mean"], "macro_f1_std": f["macro_f1_std"]})
                table.append({"name": f"{fs} | {model} | legacy thesis protocol",
                              "accuracy": lg["test"]["accuracy"],
                              "macro_f1": lg["test"]["macro_f1"]})
            detail[f"{fs} | {model}"] = {
                "per_class_recall_subject_wise": {
                    lab: grouped["pooled"]["per_class"][str(i)]["recall"]
                    for i, lab in enumerate(labels)},
                "never_predicted": [labels[int(i)] for i in grouped["pooled"]["never_predicted"]]
                if grouped["pooled"]["never_predicted"] else [],
            }
    counts = {lab: int((y == i).sum()) for i, lab in enumerate(labels)}
    return {"table": table, "detail": detail, "labels": labels, "class_counts": counts,
            "n_rows": int(len(y)), "n_subjects": int(len(np.unique(groups))),
            "keep_fraction": keep, "n_splits": n_splits}
