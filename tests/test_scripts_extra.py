import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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
