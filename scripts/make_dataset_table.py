#!/usr/bin/env python3
"""Generate the dataset tables for README.md and docs/data.md from datasets.yaml.

Usage:
    python scripts/make_dataset_table.py          # rewrite the tables in place
    python scripts/make_dataset_table.py --check  # exit 1 if the files are out of date (CI)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fel.data import load_registry, validate_registry  # noqa: E402

START = "<!-- datasets:start -->"
END = "<!-- datasets:end -->"
TARGETS = ("README.md", "docs/data.md")


def _role_text(e: dict) -> str:
    if e["role"] == "not_used":
        return "Not used"
    if e["role"] == "private_results":
        return "Private results only" if e["status"] == "done" else "Private results only (planned)"
    where = {
        "runtime": "downloaded at run time",
        "user_supplied": "you download it yourself",
        "never": "not downloaded",
    }[e["fetch"]]
    if e["status"] == "done":
        return f"Used here ({where})"
    return f"Planned ({where})"


def _cell(text: str) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def latest_check(reg: dict) -> str:
    dates = [str(e["last_checked"]) for e in reg.values() if e.get("last_checked")]
    return max(dates) if dates else "not checked"


def build_table(reg: dict) -> str:
    rows = [
        "| Dataset | What it is | Access and terms | Licence status | In this repo |",
        "|---|---|---|---|---|",
    ]
    order = {"used": 0, "private_results": 1, "not_used": 2}
    for e in sorted(reg.values(), key=lambda x: order[x["role"]]):
        access = _cell(e["access"])
        if e.get("source_url"):
            access += f" [Provider page]({e['source_url']})."
        elif e["licence_status"] == "unconfirmed":
            access += " [add provider link]"
        rows.append(
            f"| {_cell(e['title'])} | {_cell(e['what'])} | {access} "
            f"| {e['licence_status'].replace('_', ' ')} | {_role_text(e)} |"
        )
    rows.append("")
    rows.append(f"Licence information last checked: {latest_check(reg)}. "
                "Entries marked unconfirmed have not been read on the provider's page.")
    return "\n".join(rows)


def replace_block(text: str, table: str) -> str:
    if START not in text or END not in text:
        raise ValueError(f"markers {START} / {END} not found")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{table}\n{END}{tail}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    reg = load_registry()
    problems = validate_registry(reg)
    if problems:
        print("datasets.yaml problems:\n  " + "\n  ".join(problems))
        return 1
    table = build_table(reg)
    stale = []
    for rel in TARGETS:
        path = ROOT / rel
        old = path.read_text(encoding="utf-8")
        new = replace_block(old, table)
        if new != old:
            stale.append(rel)
            if not args.check:
                path.write_text(new, encoding="utf-8")
    if args.check and stale:
        print("out of date: " + ", ".join(stale) + "\nrun: python scripts/make_dataset_table.py")
        return 1
    print("dataset tables up to date" if not stale else "updated: " + ", ".join(stale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
