import numpy as np
import pytest

from fel.metrics import (
    bootstrap_ci_by_group,
    classification_metrics,
    expected_calibration_error,
    summarise_folds,
)
from fel.synthetic import make_imbalanced_labels


def test_perfect_predictions():
    y = [0, 1, 2, 0, 1, 2]
    m = classification_metrics(y, y, labels=[0, 1, 2])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["never_predicted"] == []
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_accuracy_hides_a_class_that_is_never_predicted():
    """Simulated: 92% majority class, model always predicts the majority."""
    y = make_imbalanced_labels(n=5000, majority_share=0.92, seed=0)
    pred = np.zeros_like(y)
    m = classification_metrics(y, pred, labels=list(range(6)))
    assert m["accuracy"] > 0.90
    assert m["macro_f1"] < 0.20
    assert set(m["never_predicted"]) == {"1", "2", "3", "4", "5"}
    assert m["per_class"]["0"]["recall"] == 1.0
    assert m["per_class"]["3"]["recall"] == 0.0


def test_known_per_class_values():
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1]
    m = classification_metrics(y_true, y_pred, labels=[0, 1])
    assert m["per_class"]["0"]["recall"] == pytest.approx(0.75)
    assert m["per_class"]["1"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["1"]["support"] == 2


def test_class_with_no_true_samples_has_zero_support():
    m = classification_metrics([0, 0, 1], [0, 0, 1], labels=[0, 1, 2])
    assert m["per_class"]["2"]["support"] == 0
    assert m["never_predicted"] == []


def test_length_mismatch_and_empty_raise():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0], labels=[0, 1])
    with pytest.raises(ValueError):
        classification_metrics([], [], labels=[0, 1])


def test_ece_is_zero_for_confident_correct_and_high_for_confident_wrong():
    prob = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(np.array([0, 1]), prob) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([1, 0]), prob) == pytest.approx(1.0)


def test_ece_added_to_report_when_probabilities_given():
    prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    m = classification_metrics([0, 1, 1], [0, 1, 0], labels=[0, 1], y_prob=prob)
    assert 0.0 <= m["ece"] <= 1.0


def test_group_bootstrap_interval_contains_point_estimate_and_is_repeatable():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 10)
    y = rng.integers(0, 3, size=200)
    pred = np.where(rng.random(200) < 0.7, y, rng.integers(0, 3, size=200))
    point = classification_metrics(y, pred, labels=[0, 1, 2])["accuracy"]
    lo, hi = bootstrap_ci_by_group(y, pred, groups, [0, 1, 2], metric="accuracy",
                                   n_boot=300, seed=1)
    assert lo <= point <= hi
    again = bootstrap_ci_by_group(y, pred, groups, [0, 1, 2], metric="accuracy",
                                  n_boot=300, seed=1)
    assert (lo, hi) == again


def test_bootstrap_needs_two_groups():
    with pytest.raises(ValueError):
        bootstrap_ci_by_group([0, 1], [0, 1], [5, 5], [0, 1])


def test_summarise_folds():
    folds = [{"accuracy": 0.8, "macro_f1": 0.6}, {"accuracy": 0.9, "macro_f1": 0.7}]
    s = summarise_folds(folds)
    assert s["accuracy_mean"] == pytest.approx(0.85)
    assert s["n_folds"] == 2
