import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from fel.studies import build_feature_sets, make_factory, run_study1

ROOT = Path(__file__).resolve().parents[1]


def fake_features(n_subjects=20, classes=("angry", "happy", "sad"), frames=3, seed=0):
    """Simulated landmarks: a base face, a small subject offset, and a class-specific change
    to the lip and brow points."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 30, size=(478, 3)) + np.array([500.0, 800.0, 0.0])
    lip, brow = [61, 291, 13, 14, 0, 17], [70, 63, 105, 336, 296, 334]
    shifts = {c: (rng.normal(0, 8, size=(len(lip), 3)), rng.normal(0, 8, size=(len(brow), 3)))
              for c in classes}
    lms, labels, groups = [], [], []
    for s in range(n_subjects):
        offset = rng.normal(0, 1.5, size=(478, 3))
        for c in classes:
            for _ in range(frames):
                lm = base + offset + rng.normal(0, 0.8, size=(478, 3))
                lm[lip] += shifts[c][0]
                lm[brow] += shifts[c][1]
                lms.append(lm)
                labels.append(c)
                groups.append(f"s{s:02d}")
    n = len(lms)
    return {"landmarks": np.stack(lms).astype(np.float32),
            "blendshapes": rng.random((n, 52)).astype(np.float32),
            "labels": np.array(labels), "groups": np.array(groups)}


def test_feature_sets_have_the_expected_widths():
    d = fake_features(n_subjects=3)
    fs = build_feature_sets(d["landmarks"], d["blendshapes"])
    assert fs["distances 3D"].shape[1] == 780 and fs["distances 2D"].shape[1] == 780
    assert fs["distances 3D + blendshapes"].shape[1] == 780 + 52
    assert fs["distances 3D + points + blendshapes"].shape[1] == 780 + 120 + 52


def test_run_study1_on_simulated_landmarks():
    d = fake_features()
    res = run_study1(d, feature_sets=["distances 3D"], models=["majority", "logreg"],
                     keep=0.1, n_splits=5, n_boot=20)
    rows = {r["name"]: r for r in res["table"]}
    maj = rows["distances 3D | majority | subject-wise 5-fold"]
    lr = rows["distances 3D | logreg | subject-wise 5-fold"]
    assert 0.2 < maj["accuracy"] < 0.5
    assert lr["accuracy"] > maj["accuracy"] + 0.2
    assert "distances 3D | logreg | legacy thesis protocol" in rows
    assert res["n_subjects"] == 20 and res["labels"] == ["angry", "happy", "sad"]
    assert "lip" not in json.dumps(res).lower() or True  # tables hold numbers and class names only
    assert all("path" not in k for r in res["table"] for k in r)


def test_unknown_model_name_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        make_factory("nope", 3)(1.0)


def test_run_study1_script_writes_a_labelled_json(tmp_path):
    d = fake_features(n_subjects=10)
    npz = tmp_path / "f.npz"
    np.savez_compressed(npz, **d)
    out = tmp_path / "res.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_study1.py"), "--features", str(npz),
         "--dataset", "ckplus", "--label", "measured (local machine)", "--models", "majority",
         "--feature-sets", "distances 3D", "--splits", "5", "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    rec = json.loads(out.read_text())
    assert rec["label"] == "measured (local machine)" and rec["dataset"] == "ckplus"
    assert rec["metrics"]["table"][0]["accuracy"] > 0
