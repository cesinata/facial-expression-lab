import numpy as np

from fel.imbalance import CLASS_NAMES, make_user_dataset, oversample, run_imbalance_study


def test_dataset_shape_and_majority_share():
    X, y, g, labels = make_user_dataset(n_users=40, rows_per_user=200, seed=0)
    assert X.shape == (8000, 20) and labels == CLASS_NAMES
    share = np.bincount(y).max() / len(y)
    assert 0.90 < share < 0.94
    assert len(np.unique(g)) == 40


def test_oversample_touches_only_minority_training_rows():
    y = np.array([0] * 100 + [1] * 5 + [2] * 3)
    X = np.arange(len(y), dtype=float)[:, None]
    X2, y2 = oversample(X, y, cap=20, seed=0)
    counts = np.bincount(y2)
    assert counts.tolist() == [100, 20, 20]  # the majority class (100) is left alone
    assert set(X2[y2 == 1, 0].tolist()) <= set(X[y == 1, 0].tolist())


def test_study_shows_accuracy_hiding_poor_macro_f1_on_simulated_data():
    r = run_imbalance_study(seed=0, n_users=60, rows_per_user=100)
    base = {row["name"]: row for row in r["table"]}
    assert "knn9 / class_weight" not in base
    for name in ("knn9 / baseline", "logreg / baseline", "linear_svc / baseline"):
        assert base[name]["accuracy"] >= r["majority_class_share"] - 0.03
        assert base[name]["macro_f1"] < 0.35
    assert r["split"].startswith("user-grouped")
    assert len(r["table"]) == 8
