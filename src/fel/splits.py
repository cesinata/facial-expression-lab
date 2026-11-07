"""Train, validation and test splits, with subject (group) awareness.

`random_frame_split` reproduces the protocol used in the original thesis code
(two calls to train_test_split, 70/15/15, no stratification, no groups). It exists
so the repo can show how much that protocol inflates results. Do not use it for
reported numbers.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    train_test_split,
)


def random_frame_split(
    n_samples: int,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy protocol: random split of rows, ignoring subjects and classes."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    idx = np.arange(n_samples)
    # Round the fractions: 1 - 0.7 is 0.30000000000000004 in floating point, and
    # scikit-learn rounds the test size up, which would add one extra row.
    train, rest = train_test_split(idx, test_size=round(1.0 - fractions[0], 10), random_state=seed)
    val_share = fractions[1] / (fractions[1] + fractions[2])
    val, test = train_test_split(rest, test_size=round(1.0 - val_share, 10), random_state=seed)
    return train, val, test


def group_overlap(train_idx: np.ndarray, test_idx: np.ndarray, groups: np.ndarray) -> int:
    """Number of groups (subjects) that appear in both index sets."""
    groups = np.asarray(groups)
    return len(set(groups[train_idx]) & set(groups[test_idx]))


def assert_disjoint_groups(train_idx: np.ndarray, test_idx: np.ndarray, groups: np.ndarray) -> None:
    n = group_overlap(train_idx, test_idx, groups)
    if n:
        raise ValueError(f"{n} group(s) appear in both train and test")


def grouped_kfold(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    stratified: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) with no group in both. Stratified by class when possible."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"need at least {n_splits} groups, got {n_groups}")
    if stratified:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pairs = splitter.split(np.zeros(len(y)), y, groups)
    else:
        pairs = GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups)
    for train_idx, test_idx in pairs:
        assert_disjoint_groups(train_idx, test_idx, groups)
        yield train_idx, test_idx


def leave_one_group_out(groups: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-subject-out. Use for small datasets with few subjects."""
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        raise ValueError("need at least 2 groups")
    yield from LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups)


def inner_validation_split(
    train_idx: np.ndarray,
    groups: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a training set into fit and validation parts, again by group.

    Use the validation part for early stopping and tuning, so the outer test fold
    is never touched until the final report.
    """
    train_idx = np.asarray(train_idx)
    g = np.asarray(groups)[train_idx]
    if len(np.unique(g)) < 2:
        raise ValueError("need at least 2 groups in the training set")
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    fit_rel, val_rel = next(gss.split(np.zeros(len(train_idx)), groups=g))
    return train_idx[fit_rel], train_idx[val_rel]
