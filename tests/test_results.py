import json

import pytest

from fel.results import ALLOWED_LABELS, write_result


def test_write_result_roundtrip(tmp_path):
    p = write_result(
        tmp_path / "r.json", "demo", "simulated", "synthetic",
        {"split": "grouped"}, {"accuracy": 0.5}, seed=0,
    )
    rec = json.loads(p.read_text())
    assert rec["label"] == "simulated" and rec["seed"] == 0
    assert "python" in rec["versions"] and rec["metrics"]["accuracy"] == 0.5


def test_bad_label_refused(tmp_path):
    with pytest.raises(ValueError):
        write_result(tmp_path / "r.json", "x", "great", "synthetic", {}, {}, seed=0)


def test_unknown_dataset_refused(tmp_path):
    with pytest.raises(ValueError):
        write_result(tmp_path / "r.json", "x", "simulated", "made_up_data", {}, {}, seed=0)


def test_registry_dataset_accepted(tmp_path):
    write_result(tmp_path / "r.json", "x", "not measured", "ckplus", {}, {}, seed=0)


def test_labels_include_required_ones():
    for lab in ("measured (Colab T4)", "measured (host CPU)", "measured (browser)",
                "estimated", "simulated", "not measured"):
        assert lab in ALLOWED_LABELS
