"""Temperature scaling and softmax helpers (numpy only)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y_idx: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Single temperature T > 0 minimising negative log-likelihood on a validation set.

    Fit on validation data only. Never fit on the test set.
    """
    logits = np.asarray(logits, dtype=float)
    y_idx = np.asarray(y_idx)
    if logits.ndim != 2 or len(logits) != len(y_idx):
        raise ValueError("logits must be (n, n_classes) and match y_idx")
    res = minimize_scalar(lambda t: nll(logits, y_idx, t), bounds=(0.05, 20.0),
                          method="bounded")
    return float(res.x)
