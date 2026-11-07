import numpy as np
import pytest

from fel.features import MutualInfoSelector, landmark_distances


def test_landmark_distances_3d_known_value():
    lm = np.zeros((1, 3, 3))
    lm[0, 1] = [3.0, 4.0, 0.0]
    lm[0, 2] = [0.0, 0.0, 2.0]
    d = landmark_distances(lm, [(0, 1), (0, 2)])
    assert d.shape == (1, 2)
    assert d[0, 0] == pytest.approx(5.0) and d[0, 1] == pytest.approx(2.0)


def test_landmark_distances_rejects_bad_input():
    with pytest.raises(ValueError):
        landmark_distances(np.zeros((2, 3)), [(0, 1)])
    with pytest.raises(ValueError):
        landmark_distances(np.zeros((1, 3, 2)), [(0, 5)])
    with pytest.raises(ValueError):
        landmark_distances(np.zeros((1, 3, 2)), [])


def _informative_data(seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], 100)
    informative = y[:, None] + rng.normal(0, 0.2, size=(300, 2))
    noise = rng.normal(0, 1, size=(300, 8))
    return np.hstack([noise[:, :4], informative, noise[:, 4:]]), y


def test_selector_keeps_the_informative_columns():
    X, y = _informative_data()
    sel = MutualInfoSelector(keep_fraction=0.2, random_state=0).fit(X, y)
    assert set(sel.selected_.tolist()) == {4, 5}
    assert sel.transform(X).shape == (300, 2)


def test_selector_is_repeatable_and_keeps_at_least_one():
    X, y = _informative_data()
    a = MutualInfoSelector(0.01, random_state=3).fit(X, y).selected_
    b = MutualInfoSelector(0.01, random_state=3).fit(X, y).selected_
    assert len(a) == 1 and (a == b).all()


def test_selector_uses_only_the_rows_it_is_given():
    """Fit on training rows; changing held-out labels must not change the selection."""
    X, y = _informative_data()
    train = np.arange(0, 300, 2)
    s1 = MutualInfoSelector(0.3, random_state=0).fit(X[train], y[train])
    y_changed = y.copy()
    y_changed[1::2] = 0  # held-out rows only
    s2 = MutualInfoSelector(0.3, random_state=0).fit(X[train], y_changed[train])
    assert (s1.selected_ == s2.selected_).all()


def test_selector_validates_arguments():
    X, y = _informative_data()
    with pytest.raises(ValueError):
        MutualInfoSelector(0.0).fit(X, y)
    sel = MutualInfoSelector(0.5).fit(X, y)
    with pytest.raises(ValueError):
        sel.transform(X[:, :3])
