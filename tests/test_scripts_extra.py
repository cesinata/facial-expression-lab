import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fel.results import write_result

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def _load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_script_answers_help_without_doing_any_work():
    before = sorted(p.name for p in (ROOT / "results").glob("*.json"))
    for s in SCRIPTS:
        r = subprocess.run([sys.executable, str(s), "--help"], capture_output=True, text=True,
                           cwd=ROOT, timeout=60)
        assert r.returncode == 0, f"{s.name}: {r.stderr[-300:]}"
    assert sorted(p.name for p in (ROOT / "results").glob("*.json")) == before


def test_results_table_shows_values_and_not_measured(tmp_path):
    mod = _load("make_results_table")
    write_result(tmp_path / "simulated_imbalance_study.json", "class imbalance", "simulated",
                 "synthetic", {}, {"table": [{"name": "logreg / baseline", "accuracy": 0.9,
                                              "macro_f1": 0.2, "macro_f1_std": 0.01}]}, seed=0)
    table = mod.build(tmp_path)
    row = "| class imbalance: logreg / baseline | synthetic | 0.900 | 0.200 ± 0.010 | simulated |"
    assert row in table
    assert "Study 3: FER2013 with FER+ labels | n/a | not measured" in table
    assert "reported (thesis)" in table and "reported (paper)" in table


def test_results_table_in_readme_is_up_to_date():
    mod = _load("make_results_table")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert mod.replace_block(text, mod.build(ROOT / "results")) == text


def test_percent_notebook_parsing_and_roundtrip():
    nbtools = _load("nbtools")
    text = ("# %% [markdown]\n# # Title\n# text\n\n# %%\nimport os\n# !pip install x\n"
            "# %cd repo\nx = 1\n")
    cells = nbtools.parse_percent(text)
    assert cells[0] == ("markdown", "# Title\ntext")
    assert cells[1] == ("code", "import os\n!pip install x\n%cd repo\nx = 1")
    nb = nbtools.to_ipynb(text)
    assert nb["cells"][1]["outputs"] == [] and nb["cells"][1]["execution_count"] is None


def test_notebook_pair_check_catches_outputs_and_drift(tmp_path):
    nbtools = _load("nbtools")
    py = tmp_path / "n.py"
    py.write_text("# %%\nx = 1\n")
    nb = nbtools.to_ipynb(py.read_text())
    (tmp_path / "n.ipynb").write_text(nbtools.dumps(nb))
    assert nbtools.check_pair(py) == []
    nb["cells"][0]["outputs"] = [{"output_type": "stream", "text": ["hi"]}]
    (tmp_path / "n.ipynb").write_text(json.dumps(nb))
    assert any("outputs" in p for p in nbtools.check_pair(py))
    py.write_text("# %%\nx = 2\n")
    (tmp_path / "n.ipynb").write_text(nbtools.dumps(nbtools.to_ipynb("# %%\nx = 1\n")))
    assert any("does not match" in p for p in nbtools.check_pair(py))


def test_repo_notebooks_are_in_sync_and_stripped():
    nbtools = _load("nbtools")
    pys = sorted((ROOT / "notebooks").glob("*.py"))
    assert len(pys) >= 5
    for py in pys:
        assert nbtools.check_pair(py) == [], py.name


def _study3_file(tmp_path, int8_accuracy):
    rows = [{"name": "PyTorch FP32", "accuracy": 0.70, "macro_f1": 0.57},
            {"name": "ONNX FP32", "accuracy": 0.70, "macro_f1": 0.57},
            {"name": "ONNX INT8", "accuracy": int8_accuracy, "macro_f1": 0.04}]
    write_result(tmp_path / "study3_ferplus.json", "study3", "measured (host CPU)", "ferplus",
                 {}, {"table": rows, "extra": {}}, seed=0)


def test_results_table_notes_flag_a_broken_int8_row_only_when_the_data_shows_it(tmp_path):
    mod = _load("make_results_table")
    assert mod.notes_from(tmp_path) == []
    _study3_file(tmp_path, 0.05)
    notes = mod.notes_from(tmp_path)
    assert any("no subject IDs" in n for n in notes)
    assert any("INT8" in n and "failure" in n for n in notes)
    _study3_file(tmp_path, 0.69)
    notes = mod.notes_from(tmp_path)
    assert any("no subject IDs" in n for n in notes)
    assert not any("INT8" in n for n in notes)
    assert "Note: " in mod.build(tmp_path)


def test_record_browser_latency_writes_a_labelled_result(tmp_path):
    mod = _load("record_browser_latency")
    out = tmp_path / "lat.json"
    assert mod.main(["--median-ms", "3.2", "--n-runs", "30", "--browser", "Chrome",
                     "--device", "MacBook", "--date", "2026-11-15", "--method", "live loop",
                     "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["label"] == "measured (browser)"
    lat = rec["metrics"]["extra"]["latency_browser"]
    assert (lat["median_ms"], lat["n_runs"], lat["p95_ms"]) == (3.2, 30, None)
    with pytest.raises(ValueError):
        mod.build_record(3.2, 30, "Chrome", "", "2026-11-15", "live loop")
    with pytest.raises(ValueError):
        mod.build_record(3.2, 30, "Chrome", "MacBook", "15/11/2026", "live loop")
    with pytest.raises(ValueError):
        mod.build_record(3.2, 30, "Chrome", "MacBook", "2026-11-15", "live loop", p95_ms=1.0)
