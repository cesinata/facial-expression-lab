import pytest

from fel import records as R


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")


def test_parse_kdef_name():
    d = R.parse_kdef_name("AM05SUS.JPG")
    assert d["label"] == "surprise" and d["group"] == "M05" and d["angle"] == "S"
    assert R.parse_kdef_name("BF12HAFL.jpg")["angle"] == "FL"
    assert R.parse_kdef_name("notes.txt") is None
    assert R.parse_kdef_name("AX05SUS.JPG") is None


def test_from_kdef_keeps_straight_on_and_groups_actor_across_sessions(tmp_path):
    for name in ("AF01HAS.JPG", "BF01HAS.JPG", "AF01HAFL.JPG", "AM02ANS.JPG", "readme.txt"):
        _touch(tmp_path / "x" / name)
    recs = R.from_kdef(tmp_path)
    assert len(recs) == 3
    assert {r.group for r in recs} == {"F01", "M02"}
    assert len(R.from_kdef(tmp_path, angles=("S", "FL"))) == 4


def _ck_tree(root, subject="S005", seq="001", n_frames=6, code="3.0000000e+00"):
    for i in range(1, n_frames + 1):
        _touch(root / "img" / subject / seq / f"{subject}_{seq}_{i:08d}.png")
    e = root / "emo" / subject / seq / f"{subject}_{seq}_{n_frames:08d}_emotion.txt"
    e.parent.mkdir(parents=True, exist_ok=True)
    e.write_text(f"   {code}\n")


def test_from_ckplus_thesis_frames(tmp_path):
    _ck_tree(tmp_path)
    recs = R.from_ckplus(tmp_path / "img", tmp_path / "emo")
    assert [r.label for r in recs] == ["disgust"] * 3 + ["neutral"]
    assert {r.group for r in recs} == {"S005"}
    assert recs[-1].path.name.endswith("00000001.png")


def test_from_ckplus_peak3_and_skips(tmp_path):
    _ck_tree(tmp_path, "S005", "001")
    _ck_tree(tmp_path, "S006", "001", n_frames=3)           # too few frames
    _ck_tree(tmp_path, "S007", "001", code="not a number")  # unreadable label
    _touch(tmp_path / "img" / "S008" / "001" / "S008_001_00000001.png")  # no emotion file
    recs = R.from_ckplus(tmp_path / "img", tmp_path / "emo", frames="peak3")
    assert len(recs) == 3 and {r.label for r in recs} == {"disgust"}
    with pytest.raises(ValueError):
        R.from_ckplus(tmp_path / "img", tmp_path / "emo", frames="all")


def test_from_regex_with_label_map(tmp_path):
    for name in ("char01/joy_light1.png", "char02/anger_light2.png", "char02/skip.txt"):
        _touch(tmp_path / name)
    recs = R.from_regex(tmp_path, r"/(\w+?)_light\d\.png$", r"^/(char\d+)/",
                        label_map={"joy": "happy"})
    assert sorted((r.label, r.group) for r in recs) == [("anger", "char02"), ("happy", "char01")]
