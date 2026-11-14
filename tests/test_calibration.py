import numpy as np
import pytest

from fel.calibration import fit_temperature, nll, softmax


def test_softmax_rows_sum_to_one_and_respect_temperature():
    z = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    p = softmax(z)
    assert np.allclose(p.sum(1), 1)
    assert softmax(z, 10.0)[0].max() < p[0].max()


def test_fit_temperature_cools_overconfident_logits():
    rng = np.random.default_rng(0)
    n, k = 2000, 4
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.0     # informative but noisy
    overconfident = logits * 5.0
    t = fit_temperature(overconfident, y)
    assert t > 2.0
    assert nll(overconfident, y, t) < nll(overconfident, y, 1.0)


def test_fit_temperature_validates_shapes():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((3, 2)), np.zeros(4, dtype=int))
