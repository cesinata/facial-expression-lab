import numpy as np
import pytest

from fel import landmarks as L


def _fake_landmarks(n=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 50, size=(n, L.N_LANDMARKS, 3)) + np.array([500.0, 800.0, 0.0])


def test_pair_count_and_order_are_fixed():
    pts = L.all_points()
    assert len(pts) == 40
    pairs = L.build_pairs()
    assert len(pairs) == 40 * 39 // 2 == 780
    assert pairs == L.build_pairs()
    assert len(L.feature_names()) == 780


def test_every_selected_point_has_one_region_and_pairs_are_tagged():
    for p in L.all_points():
        L.region_of_point(p)
    assert L.pair_region(61, 291) == "lip"
    assert L.pair_region(61, 152) == "cross"
    with pytest.raises(KeyError):
        L.region_of_point(400)


def test_distances_ignore_scale_and_translation():
    lm = _fake_landmarks()
    d1 = L.distance_features(lm)
    d2 = L.distance_features(lm * 3.7 + np.array([10.0, -20.0, 5.0]))
    assert d1.shape == (3, 780)
    assert np.allclose(d1, d2)


def test_2d_distances_differ_from_3d():
    lm = _fake_landmarks()
    d3, d2 = L.distance_features(lm, use_z=True), L.distance_features(lm, use_z=False)
    assert not np.allclose(d3, d2)


def test_distance_features_reject_bad_input():
    with pytest.raises(ValueError):
        L.distance_features(np.zeros((2, 10, 3)))
    lm = _fake_landmarks(1)
    lm[0, L.OUTER_EYE_LEFT] = lm[0, L.OUTER_EYE_RIGHT]
    with pytest.raises(ValueError):
        L.distance_features(lm)


def test_point_features_are_centred_on_the_nose_tip():
    lm = _fake_landmarks(2)
    pf = L.point_features(lm)
    assert pf.shape == (2, 40 * 3)
    pts = L.all_points()
    nose_col = pts.index(1) * 3
    assert np.allclose(pf[:, nose_col : nose_col + 3], 0.0)


def test_to_pixel_scale():
    lm = np.ones((1, 2, 3)) * 0.5
    out = L.to_pixel_scale(lm, width=1000, height=2000)
    assert np.allclose(out[0, 0], [500, 1000, 500])


def test_face_box_is_clipped_to_the_image():
    lm = np.array([[10.0, 10.0, 0], [90.0, 90.0, 0]])
    assert L.face_box(lm, 100, 100, margin=0.5) == (0, 0, 100, 100)
    assert L.face_box(lm, 100, 100, margin=0.0) == (10, 10, 90, 90)


def test_extract_many_skips_images_without_a_face():
    lm = _fake_landmarks(1)[0]
    bs = np.zeros(L.N_BLENDSHAPES)

    def fake(path):
        return None if path == "none.png" else (lm, bs, (0, 0, 10, 10))

    ok, lms, bss, boxes = L.extract_many(["a.png", "none.png", "b.png"], fake)
    assert ok.tolist() == [True, False, True]
    assert lms.shape == (2, L.N_LANDMARKS, 3) and bss.shape == (2, 52) and boxes.shape == (2, 4)
    ok, lms, bss, boxes = L.extract_many(["none.png"], fake)
    assert lms.shape[0] == 0
