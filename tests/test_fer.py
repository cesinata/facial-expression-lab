import numpy as np
import pandas as pd
import pytest

from fel import fer


def test_ferplus_majority_rules():
    #            neu hap sur sad ang dis fea con unk NF
    votes = np.array([
        [8, 1, 0, 0, 0, 0, 0, 0, 1, 0],   # clear neutral
        [1, 5, 4, 0, 0, 0, 0, 0, 0, 0],   # happiness
        [3, 3, 0, 0, 0, 0, 0, 0, 2, 2],   # tie at the top: dropped
        [1, 1, 0, 0, 0, 0, 0, 0, 3, 5],   # unknown + NF outnumber: dropped
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10],  # no emotion votes: dropped
        [0, 0, 0, 0, 0, 0, 0, 7, 1, 2],   # contempt
    ])
    assert fer.ferplus_majority(votes).tolist() == [0, 1, -1, -1, -1, 7]


def test_load_fer2013_csv(tmp_path):
    pix = " ".join(["10"] * 2304)
    pd.DataFrame({"emotion": [0, 1], "pixels": [pix, pix],
                  "Usage": ["Training", "PrivateTest"]}).to_csv(tmp_path / "f.csv", index=False)
    x, usage = fer.load_fer2013_csv(str(tmp_path / "f.csv"))
    assert x.shape == (2, 48, 48) and x.dtype == np.uint8
    parts = fer.split_by_usage(usage)
    assert parts["Training"].tolist() == [0] and parts["PrivateTest"].tolist() == [1]
    assert parts["PublicTest"].size == 0


def test_load_fer2013_csv_rejects_wrong_pixel_count(tmp_path):
    df = pd.DataFrame({"pixels": ["1 2 3"], "Usage": ["Training"]})
    df.to_csv(tmp_path / "f.csv", index=False)
    with pytest.raises(ValueError):
        fer.load_fer2013_csv(str(tmp_path / "f.csv"))


def test_load_ferplus_votes_checks_columns(tmp_path):
    cols = fer.FERPLUS_CLASSES + fer.FERPLUS_EXTRA
    pd.DataFrame([[1] * 10], columns=["Usage", "Image name"] + cols[:8]).to_csv(
        tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        fer.load_ferplus_votes(str(tmp_path / "bad.csv"))
    pd.DataFrame([[2] * 10], columns=cols).to_csv(tmp_path / "ok.csv", index=False)
    assert fer.load_ferplus_votes(str(tmp_path / "ok.csv")).shape == (1, 10)


def test_shared_emotion_map_has_six_entries():
    assert len(fer.SHARED_WITH_SIX) == 6 and "contempt" not in fer.SHARED_WITH_SIX


def test_icml_style_columns_with_leading_spaces_are_accepted(tmp_path):
    pix = " ".join(["7"] * 2304)
    pd.DataFrame({"emotion": [3], " Usage": ["PublicTest"], " pixels": [pix]}).to_csv(
        tmp_path / "icml_face_data.csv", index=False)
    x, usage = fer.load_fer2013_csv(str(tmp_path / "icml_face_data.csv"))
    assert x.shape == (1, 48, 48) and usage.tolist() == ["PublicTest"]


def test_study3_script_checks_that_fer_and_ferplus_rows_line_up():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "study3.py").read_text()
    assert "load_ferplus_usage" in src and "usage_mismatch_fraction" in src


def test_usage_alignment_helpers(tmp_path):
    cols = fer.FERPLUS_CLASSES + fer.FERPLUS_EXTRA
    df = pd.DataFrame([[1] * 10, [2] * 10], columns=cols)
    df.insert(0, "Usage", ["Training", " PrivateTest "])
    df.to_csv(tmp_path / "plus.csv", index=False)
    u = fer.load_ferplus_usage(str(tmp_path / "plus.csv"))
    assert u.tolist() == ["Training", "PrivateTest"]
    assert fer.usage_mismatch_fraction(u, np.array(["Training", "PrivateTest"])) == 0.0
    assert fer.usage_mismatch_fraction(u, np.array(["Training", "Training"])) == 0.5
    with pytest.raises(ValueError):
        fer.usage_mismatch_fraction(u, np.array(["Training"]))
    pd.DataFrame([[1] * 10], columns=cols).to_csv(tmp_path / "nousage.csv", index=False)
    assert fer.load_ferplus_usage(str(tmp_path / "nousage.csv")) is None
