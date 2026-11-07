import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from fel.splits import (
    assert_disjoint_groups,
    group_overlap,
    grouped_kfold,
    inner_validation_split,
    leave_one_group_out,
    random_frame_split,
)
from fel.synthetic import make_sequence_dataset


def test_random_frame_split_sizes_and_disjoint():
    tr, va, te = random_frame_split(2320, seed=42)
    assert len(tr) == 1624 and len(va) == 348 and len(te) == 348
    assert len(set(tr) & set(va)) == len(set(tr) & set(te)) == len(set(va) & set(te)) == 0
    assert len(set(tr) | set(va) | set(te)) == 2320


def test_random_frame_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        random_frame_split(100, fractions=(0.5, 0.2, 0.2))


def test_random_split_leaks_subjects_but_grouped_does_not():
    X, y, g = make_sequence_dataset(seed=1)
    tr, _, te = random_frame_split(len(y), seed=42)
    assert group_overlap(tr, te, g) > 0
    for a, b in grouped_kfold(y, g, n_splits=5, seed=0):
        assert group_overlap(a, b, g) == 0


def test_grouped_kfold_covers_every_row_once():
    X, y, g = make_sequence_dataset(seed=2)
    seen = np.concatenate([b for _, b in grouped_kfold(y, g, n_splits=5, seed=0)])
    assert sorted(seen.tolist()) == list(range(len(y)))


def test_grouped_kfold_needs_enough_groups():
    y = np.array([0, 1, 0, 1])
    g = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        list(grouped_kfold(y, g, n_splits=5))


def test_assert_disjoint_groups_raises_on_overlap():
    g = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        assert_disjoint_groups(np.array([0, 2]), np.array([1, 3]), g)


def test_leave_one_group_out_holds_out_each_group():
    g = np.array([0, 0, 1, 1, 2, 2])
    folds = list(leave_one_group_out(g))
    assert len(folds) == 3
    for tr, te in folds:
        assert len(set(g[te])) == 1 and group_overlap(tr, te, g) == 0


def test_inner_validation_split_is_group_aware():
    X, y, g = make_sequence_dataset(seed=3)
    tr, _ = next(grouped_kfold(y, g, 5, seed=0))
    fit, val = inner_validation_split(tr, g, val_fraction=0.2, seed=0)
    assert group_overlap(fit, val, g) == 0
    assert set(fit) | set(val) == set(tr)


def test_random_frame_protocol_inflates_accuracy_on_simulated_sequences():
    """Simulated data: sibling frames make a random split look far better than a grouped one."""
    gaps = []
    for seed in range(3):
        X, y, g = make_sequence_dataset(seed=seed)
        tr, _, te = random_frame_split(len(y), seed=42)
        acc_random = KNeighborsClassifier(1).fit(X[tr], y[tr]).score(X[te], y[te])
        accs = [
            KNeighborsClassifier(1).fit(X[a], y[a]).score(X[b], y[b])
            for a, b in grouped_kfold(y, g, 5, seed=0)
        ]
        gaps.append(acc_random - float(np.mean(accs)))
    assert min(gaps) > 0.25
