#!/usr/bin/env python3
"""Write notebooks/*.ipynb from notebooks/*.py (percent format).

    python scripts/sync_notebooks.py           # write
    python scripts/sync_notebooks.py --check   # exit 1 if a pair is out of sync or has outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nbtools  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true")
    check = ap.parse_args(argv).check
    problems = []
    for py in sorted((ROOT / "notebooks").glob("*.py")):
        if check:
            problems += nbtools.check_pair(py)
        else:
            py.with_suffix(".ipynb").write_text(nbtools.dumps(nbtools.to_ipynb(py.read_text())),
                                               encoding="utf-8")
    if problems:
        print("\n".join(problems))
        return 1
    print("notebooks in sync" if check else "notebooks written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
