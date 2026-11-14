#!/usr/bin/env python3
"""Unpack the Kaggle FER2013 download and the FERPlus GitHub zip, find the files Study 3 needs,
and print the FER+ licence so you can read it.

    python scripts/prepare_fer_data.py --kaggle-file fer2013.tar.gz \\
        --ferplus-zip FERPlus-master.zip --out-dir /content/fer_data

Writes OUT_DIR/paths.json with fer_csv, ferplus_csv and license_path. The Kaggle zip may hold
fer2013.csv, a fer2013.tar.gz, or icml_face_data.csv (also inside a zip). All are handled.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path

FER_NAMES = ("fer2013.csv", "icml_face_data.csv")


def _extract_zip(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)


def _extract_tar(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as t:
        try:
            t.extractall(dest, filter="data")
        except TypeError:  # Python versions without the filter argument
            t.extractall(dest)


def unpack_nested(root: Path, max_rounds: int = 3) -> None:
    """Extract every .zip and .tar.gz found under root, a few levels deep."""
    done: set[Path] = set()
    for _ in range(max_rounds):
        found = [p for p in root.rglob("*") if p.is_file() and p not in done
                 and (p.suffix.lower() == ".zip" or p.name.lower().endswith((".tar.gz", ".tgz")))]
        if not found:
            return
        for p in found:
            done.add(p)
            if p.suffix.lower() == ".zip":
                _extract_zip(p, p.parent / (p.stem + "_unzipped"))
            else:
                _extract_tar(p, p.parent / (p.name.split(".")[0] + "_untarred"))


def find_file(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:  # earlier names win
        hits = sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower() == name.lower())
        if hits:
            return hits[0]
    return None


def prepare(kaggle_file: Path, ferplus_zip: Path, out_dir: Path) -> dict[str, str]:
    """kaggle_file may be fer2013.zip or fer2013.tar.gz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k_dir, p_dir = out_dir / "kaggle", out_dir / "ferplus"
    if kaggle_file.name.lower().endswith((".tar.gz", ".tgz")):
        _extract_tar(kaggle_file, k_dir)
    else:
        _extract_zip(kaggle_file, k_dir)
    unpack_nested(k_dir)
    _extract_zip(ferplus_zip, p_dir)
    fer = find_file(k_dir, FER_NAMES)
    plus = find_file(p_dir, ("fer2013new.csv",))
    lic = find_file(p_dir, ("LICENSE.md", "LICENSE", "LICENSE.txt"))
    missing = [n for n, v in (("FER2013 csv", fer), ("fer2013new.csv", plus)) if v is None]
    if missing:
        files = sorted(p for p in out_dir.rglob("*") if p.is_file())
        listing = [str(p.relative_to(out_dir)) for p in files]
        n_img = sum(1 for p in files if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        hint = ""
        if n_img > 100 and "FER2013 csv" in missing:
            hint = (f" The Kaggle zip holds {n_img} image files and no CSV. That looks like an "
                    "image-folder copy of FER2013. FER+ labels need the original fer2013.csv "
                    "(row numbers and the Usage column). Download fer2013.tar.gz from the "
                    "official competition page.")
        raise FileNotFoundError(f"could not find: {', '.join(missing)}.{hint} "
                                f"Files seen (first 40): {listing[:40]}")
    paths = {"fer_csv": str(fer), "ferplus_csv": str(plus), "license_path": str(lic) if lic else ""}
    (out_dir / "paths.json").write_text(json.dumps(paths, indent=1))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--kaggle-file", "--kaggle-zip", dest="kaggle_file", required=True,
                    help="fer2013.tar.gz or fer2013.zip from the Kaggle competition page")
    ap.add_argument("--ferplus-zip", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    try:
        paths = prepare(Path(a.kaggle_file), Path(a.ferplus_zip), Path(a.out_dir))
    except (FileNotFoundError, zipfile.BadZipFile, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(paths, indent=1))
    if paths["license_path"]:
        print("\n===== FER+ licence file (read it, and say what it allows) =====")
        print("\n".join(Path(paths["license_path"]).read_text(errors="replace").splitlines()[:60]))
    else:
        print("\nNo LICENSE file found in the FER+ zip. Read the repository page instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
