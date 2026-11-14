"""Convert jupytext 'percent' .py files to .ipynb and check that pairs are in sync.

Rules: a cell starts at a line beginning '# %%' ('# %% [markdown]' for text). In markdown
cells each line starts with '# '. In code cells, lines like '# !pip ...' and '# %run ...' are
shell or magic commands and become '!pip ...' and '%run ...' in the notebook.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MAGIC = re.compile(r"^# (?=[!%])")


def parse_percent(text: str) -> list[tuple[str, str]]:
    cells: list[list] = []
    for line in text.splitlines():
        if line.startswith("# %%"):
            cells.append(["markdown" if "[markdown]" in line else "code", []])
        elif cells:
            cells[-1][1].append(line)
    out = []
    for kind, lines in cells:
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            lines = [re.sub(r"^# ?", "", ln) for ln in lines]
        else:
            lines = [MAGIC.sub("", ln) for ln in lines]
        out.append((kind, "\n".join(lines)))
    return out


def _source(text: str) -> list[str]:
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if text else []


def to_ipynb(text: str) -> dict:
    cells = []
    for kind, src in parse_percent(text):
        cell = {"cell_type": kind, "metadata": {}, "source": _source(src)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def dumps(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def check_pair(py_path: Path) -> list[str]:
    """Problems for one notebook pair (empty list means fine)."""
    ipynb = py_path.with_suffix(".ipynb")
    if not ipynb.exists():
        return [f"{ipynb} is missing (run: python scripts/sync_notebooks.py)"]
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    problems = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "code" and (c.get("outputs") or c.get("execution_count")):
            problems.append(f"{ipynb}: cell {i} has outputs or an execution count; strip them")
    want = [(c["cell_type"], "".join(c["source"])) for c in to_ipynb(py_path.read_text())["cells"]]
    have = [(c["cell_type"], "".join(c["source"])) for c in nb.get("cells", [])]
    if want != have:
        problems.append(f"{ipynb} does not match {py_path} (run: python scripts/sync_notebooks.py)")
    return problems


if __name__ == "__main__":
    print("This is a helper module. Use scripts/sync_notebooks.py.")
