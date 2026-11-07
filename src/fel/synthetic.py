"""Synthetic data for tests and for demonstrating evaluation leakage.

Each subject performs each expression once as a short sequence of near-identical
frames, like the first and last frames of a video clip. Frames from one sequence
share a sequence-specific offset. A random frame split puts sibling frames in
train and test, so accuracy looks high. A subject-wise split does not, so
accuracy drops. All data here is simulated.
"""

from __future__ import annotations

import numpy as np


def make_sequence_dataset(
    n_subjects: int = 30,
    n_classes: int = 4,
    frames_per_sequence: int = 4,
    n_features: int = 12,
    class_scale: float = 0.6,
    sequence_scale: float = 1.5,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n, n_features), y (n,), groups (n,) with one group per subject."""
    rng = np.random.default_rng(seed)
    class_means = rng.normal(0.0, class_scale, size=(n_classes, n_features))
    X, y, g = [], [], []
    for s in range(n_subjects):
        for c in range(n_classes):
            centre = class_means[c] + rng.normal(0.0, sequence_scale, size=n_features)
            frames = centre + rng.normal(0.0, noise, size=(frames_per_sequence, n_features))
            X.append(frames)
            y.extend([c] * frames_per_sequence)
            g.extend([s] * frames_per_sequence)
    return np.vstack(X), np.array(y), np.array(g)


def make_imbalanced_labels(
    n: int = 2000,
    majority_share: float = 0.92,
    n_minor: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Labels with one dominant class (0) and `n_minor` small classes. Simulated."""
    if not (0.0 < majority_share < 1.0):
        raise ValueError("majority_share must be in (0, 1)")
    rng = np.random.default_rng(seed)
    minor = (1.0 - majority_share) / n_minor
    probs = [majority_share] + [minor] * n_minor
    return rng.choice(n_minor + 1, size=n, p=probs)
