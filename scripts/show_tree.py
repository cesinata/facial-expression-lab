#!/usr/bin/env python3
"""Show what a folder, a .zip or a .tar.gz contains, without extracting anything.

    python scripts/show_tree.py PATH [--depth 3] [--max-items 12]

Prints file counts by extension, top-level entries, nested archives and a few file NAMES.
Never prints file contents. Use it to check what a download really holds.
"""

from __future__ import annotations

import argparse
import collections
import tarfile
import zipfile
from pathlib import Path

ARCHIVE_ENDINGS = (".zip", ".tar.gz", ".tgz", ".tar")


def _ext(name: str) -> str:
    low = name.lower()
    for e in (".tar.gz",):
        if low.endswith(e):
            return e
    return Path(low).suffix or "(none)"


def summarise_names(names: list[str], sizes: list[int] | None = None, max_items: int = 12,
                    indent: str = "") -> list[str]:
    files = [n for n in names if not n.endswith("/")]
    ext = collections.Counter(_ext(n) for n in files)
    top = collections.Counter(n.strip("/").split("/")[0] for n in names if n.strip("/"))
    lines = [f"{indent}files: {len(files)}"
             + (f", about {sum(sizes) / 1e6:.1f} MB uncompressed" if sizes else "")]
    lines.append(f"{indent}by extension: {dict(ext.most_common(8))}")
    lines.append(f"{indent}top-level entries ({len(top)}): {list(top)[:max_items]}")
    nested = [n for n in files if n.lower().endswith(ARCHIVE_ENDINGS)]
    if nested:
        lines.append(f"{indent}nested archives: {nested[:max_items]}")
    step = max(1, len(files) // max_items)
    lines.append(f"{indent}example file names: {files[::step][:max_items]}")
    return lines


def describe_archive(path: Path, max_items: int = 12, indent: str = "") -> list[str]:
    lines = [f"{indent}[archive] {path.name}"]
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                infos = z.infolist()
                lines += summarise_names([i.filename for i in infos],
                                         [i.file_size for i in infos if not i.is_dir()],
                                         max_items, indent + "  ")
        else:
            with tarfile.open(path) as t:
                members = t.getmembers()
                names = [m.name + ("/" if m.isdir() else "") for m in members]
                lines += summarise_names(names, [m.size for m in members if m.isfile()],
                                         max_items, indent + "  ")
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        lines.append(f"{indent}  could not open: {exc}")
    return lines


def describe_dir(root: Path, depth: int = 3, max_items: int = 12, _level: int = 0) -> list[str]:
    indent = "  " * _level
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file() and not p.name.startswith(".")]
    n_files_here = len(files)
    lines = [f"{indent}{root.name}/  ({len(dirs)} folders, {n_files_here} files here)"]
    for f in files[:max_items]:
        if f.name.lower().endswith(ARCHIVE_ENDINGS):
            lines += describe_archive(f, max_items, indent + "  ")
        else:
            lines.append(f"{indent}  {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")
    if n_files_here > max_items:
        by_ext = collections.Counter(_ext(f.name) for f in files[max_items:])
        lines.append(f"{indent}  ... and {n_files_here - max_items} more files {dict(by_ext)}")
    if _level + 1 < depth:
        for d in dirs[:max_items]:
            lines += describe_dir(d, depth, max_items, _level + 1)
    elif dirs:
        lines.append(f"{indent}  (folders below depth {depth} not shown: "
                     f"{[d.name for d in dirs[:max_items]]})")
    if len(dirs) > max_items:
        lines.append(f"{indent}  ... and {len(dirs) - max_items} more folders")
    return lines


def describe(path: Path, depth: int = 3, max_items: int = 12) -> list[str]:
    if path.is_dir():
        return describe_dir(path, depth, max_items)
    if path.is_file() and path.name.lower().endswith(ARCHIVE_ENDINGS):
        return describe_archive(path, max_items)
    if path.is_file():
        return [f"{path.name}  ({path.stat().st_size / 1e6:.2f} MB)"]
    return [f"not found: {path}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max-items", type=int, default=12)
    a = ap.parse_args()
    print("\n".join(describe(Path(a.path).expanduser(), a.depth, a.max_items)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
