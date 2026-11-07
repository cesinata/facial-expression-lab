import importlib.util
import json
from pathlib import Path

from fel.data import load_registry

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registry_check_passes_on_allowed_dataset(tmp_path):
    mod = _load("check_notebooks_registry")
    f = tmp_path / "ok.py"
    f.write_text('from fel.data import download\ndownload("uibvfedplus_light")\n')
    assert mod.check([f], load_registry()) == []


def test_registry_check_fails_on_unknown_private_and_user_supplied(tmp_path):
    mod = _load("check_notebooks_registry")
    f = tmp_path / "bad.py"
    f.write_text('download("nope")\ndownload("ckplus")\ndownload("jaffe")\n')
    problems = mod.check([f], load_registry())
    assert len(problems) == 3


def test_registry_check_reads_notebooks(tmp_path):
    mod = _load("check_notebooks_registry")
    nb = {"cells": [{"cell_type": "code", "source": ['download("kdef")']}]}
    f = tmp_path / "n.ipynb"
    f.write_text(json.dumps(nb))
    assert len(mod.check([f], load_registry())) == 1


def test_dataset_tables_in_repo_are_up_to_date():
    mod = _load("make_dataset_table")
    table = mod.build_table(load_registry())
    for rel in mod.TARGETS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert mod.replace_block(text, table) == text, f"{rel} is out of date"


def test_table_marks_jaffe_as_user_supplied_and_private_as_private():
    mod = _load("make_dataset_table")
    table = mod.build_table(load_registry())
    jaffe = next(line for line in table.splitlines() if line.startswith("| JAFFE"))
    assert "you download it yourself" in jaffe
    ck = next(line for line in table.splitlines() if line.startswith("| CK+"))
    assert "Private results only" in ck
