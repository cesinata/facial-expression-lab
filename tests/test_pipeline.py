import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from fel.pipeline import (
    PreprocessedModel,
    SklearnAdapter,
    encode_labels,
    run_cv,
    run_legacy_protocol,
)
from fel.synthetic import make_sequence_dataset


def _knn(frac):
    return PreprocessedModel(SklearnAdapter(KNeighborsClassifier(1), 4), frac)


def _logreg(frac):
    return PreprocessedModel(SklearnAdapter(LogisticRegression(max_iter=300), 4), frac)


def test_adapter_expands_missing_classes():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.array([0, 2] * 10)
    a = SklearnAdapter(LogisticRegression(), n_classes=4).fit(X, y)
    p = a.predict_proba(X)
    assert p.shape == (20, 4) and np.allclose(p[:, [1, 3]], 0) and np.allclose(p.sum(1), 1)


def test_encode_labels_is_sorted_and_stable():
    y, labels = encode_labels(["sad", "angry", "sad", "happy"])
    assert labels == ["angry", "happy", "sad"] and y.tolist() == [2, 0, 2, 1]


def test_grouped_cv_shares_no_subjects_and_random_cv_does():
    X, y, g = make_sequence_dataset(seed=0)
    labels = ["a", "b", "c", "d"]
    grouped = run_cv(X, y, g, labels, _logreg, n_splits=5, grouped=True, n_boot=50)
    rand = run_cv(X, y, g, labels, _logreg, n_splits=5, grouped=False, n_boot=50)
    assert grouped["mean_groups_shared_between_train_and_test"] == 0
    assert rand["mean_groups_shared_between_train_and_test"] > 0
    assert grouped["protocol"] == "grouped k-fold"
    lo, hi = grouped["macro_f1_ci95_over_subjects"]
    assert lo <= hi


def test_random_cv_and_legacy_protocol_inflate_results_on_simulated_sequences():
    X, y, g = make_sequence_dataset(seed=1)
    labels = ["a", "b", "c", "d"]
    grouped = run_cv(X, y, g, labels, _knn, n_splits=5, grouped=True, n_boot=50)
    rand = run_cv(X, y, g, labels, _knn, n_splits=5, grouped=False, n_boot=50)
    legacy = run_legacy_protocol(X, y, g, labels, _knn, keep_fraction=1.0)
    assert rand["folds"]["macro_f1_mean"] - grouped["folds"]["macro_f1_mean"] > 0.25
    assert legacy["test"]["macro_f1"] - grouped["folds"]["macro_f1_mean"] > 0.25
    assert legacy["groups_shared_between_train_and_test"] > 0
