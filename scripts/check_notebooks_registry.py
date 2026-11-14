#!/usr/bin/env python3
"""Fail if a notebook or script uses a dataset that datasets.yaml does not allow.

Looks for download("name") and download('name') in .ipynb and .py files under
notebooks/ and scripts/ (and any path given on the command line). A name must exist
in the registry and have fetch: runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fel.data import load_registry  # noqa: E402

PATTERN = re.compile(r"""download\(\s*['"]([A-Za-z0-9_\-]+)['"]""")


def source_of(path: Path) -> str:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(c.get("source", "")) for c in nb.get("cells", [])
                         if c.get("cell_type") == "code")
    return path.read_text(encoding="utf-8")


def check(paths: list[Path], registry: dict) -> list[str]:
    problems = []
    for p in paths:
        if p.name == Path(__file__).name:
            continue
        for name in PATTERN.findall(source_of(p)):
            if name not in registry:
                problems.append(f"{p}: dataset '{name}' is not in datasets.yaml")
            elif registry[name]["fetch"] != "runtime":
                problems.append(
                    f"{p}: dataset '{name}' has fetch '{registry[name]['fetch']}' "
                    "and must not be downloaded by code"
                )
    return problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="files or folders (default: notebooks/ and scripts/)")
    argv = ap.parse_args(argv).paths
    roots = [Path(a) for a in argv] or [ROOT / "notebooks", ROOT / "scripts"]
    files: list[Path] = []
    for r in roots:
        if r.is_file():
            files.append(r)
        elif r.exists():
            files += sorted(list(r.rglob("*.ipynb")) + list(r.rglob("*.py")))
    problems = check(files, load_registry())
    if problems:
        print("\n".join(problems))
        return 1
    print(f"registry check passed ({len(files)} files scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
