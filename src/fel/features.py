"""Landmark distance features and Mutual Information selection.

The selector must be fitted on training rows only. In the original thesis code,
Mutual Information was computed on all rows before the split, which lets test
labels influence which features are kept.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


def landmark_distances(landmarks: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Euclidean distance between landmark pairs.

    landmarks: (n_samples, n_landmarks, n_dims) with n_dims 2 or 3.
    pairs: list of (i, j) landmark indices. Returns (n_samples, len(pairs)).
    """
    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.ndim != 3:
        raise ValueError("landmarks must have shape (n_samples, n_landmarks, n_dims)")
    if not pairs:
        raise ValueError("pairs is empty")
    n_lm = landmarks.shape[1]
    for i, j in pairs:
        if not (0 <= i < n_lm and 0 <= j < n_lm):
            raise ValueError(f"pair ({i}, {j}) is outside 0..{n_lm - 1}")
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return np.linalg.norm(landmarks[:, a, :] - landmarks[:, b, :], axis=2)


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """Keep the top fraction of features ranked by Mutual Information with the label.

    keep_fraction=0.25 keeps the top 25% (rounded up, at least one feature).
    Ties are broken by feature order, so results are repeatable.
    """

    def __init__(self, keep_fraction: float = 0.25, random_state: int = 0):
        self.keep_fraction = keep_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if not (0.0 < self.keep_fraction <= 1.0):
            raise ValueError("keep_fraction must be in (0, 1]")
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and match y in length")
        if self.keep_fraction >= 1.0:  # keep everything: skip the slow estimate
            self.scores_ = np.zeros(X.shape[1])
            self.selected_ = np.arange(X.shape[1])
            self.n_features_in_ = X.shape[1]
            return self
        self.scores_ = mutual_info_classif(X, y, random_state=self.random_state)
        k = max(1, math.ceil(self.keep_fraction * X.shape[1]))
        order = np.argsort(-self.scores_, kind="stable")
        self.selected_ = np.sort(order[:k])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than at fit time")
        return X[:, self.selected_]
