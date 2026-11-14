#!/usr/bin/env python3
"""Print the structure of a dataset folder or file so the right loader settings can be chosen.

    python scripts/probe_dataset.py folder PATH          any image folder (UIBVFEDPlus-Light etc.)
    python scripts/probe_dataset.py kdef PATH
    python scripts/probe_dataset.py ckplus IMAGES_DIR EMOTION_DIR
    python scripts/probe_dataset.py jaffe PATH
    python scripts/probe_dataset.py fer FER2013_CSV [FERPLUS_CSV]

Prints counts and a few file NAMES (never images, never contents). Paste the output back
if you want help choosing settings.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fel import records as R  # noqa: E402


def probe_folder(path: Path) -> None:
    files = [p for p in path.rglob("*") if p.is_file()]
    ext = collections.Counter(p.suffix.lower() for p in files)
    depth = collections.Counter(len(p.relative_to(path).parts) for p in files)
    print(f"files: {len(files)}")
    print("extensions:", dict(ext.most_common(8)))
    print("files by folder depth:", dict(sorted(depth.items())))
    top = collections.Counter(p.relative_to(path).parts[0] for p in files)
    print("top-level entries (first 15):", dict(list(top.items())[:15]))
    print("example relative paths:")
    step = max(1, len(files) // 12)
    for p in sorted(files)[::step][:12]:
        print("  ", p.relative_to(path).as_posix())
    total = sum(p.stat().st_size for p in files)
    print(f"total size: {total / 1e9:.2f} GB")


def summarise(recs: list) -> None:
    print(f"records: {len(recs)}")
    print("labels:", dict(collections.Counter(r.label for r in recs)))
    print("groups (subjects):", len({r.group for r in recs}))
    for r in recs[:5]:
        print("  example:", r.path.name, "->", r.label, "/", r.group)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("kind", choices=["folder", "kdef", "ckplus", "jaffe", "fer"])
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    p = [Path(x) for x in a.paths]
    if a.kind == "folder":
        probe_folder(p[0])
    elif a.kind == "kdef":
        summarise(R.from_kdef(p[0]))
    elif a.kind == "jaffe":
        summarise(R.from_jaffe(p[0]))
    elif a.kind == "ckplus":
        if len(p) < 2:
            ap.error("ckplus needs IMAGES_DIR and EMOTION_DIR")
        summarise(R.from_ckplus(p[0], p[1]))
    else:
        import pandas as pd

        for f in p:
            df = pd.read_csv(f, nrows=3)
            print(f.name, "columns:", list(df.columns))
            print(f.name, "rows:", sum(1 for _ in open(f, encoding="utf-8")) - 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
