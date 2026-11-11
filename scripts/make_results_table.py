#!/usr/bin/env python3
"""Fill the results table in README.md from results/*.json (and results/private/*.json).

    python scripts/make_results_table.py           # rewrite the table
    python scripts/make_results_table.py --check   # exit 1 if README is out of date (CI)

Nothing is typed by hand: a number appears only if a result file holds it. Expected results
that have no file yet show "not measured". Reported numbers from older work are listed with
their own label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"

REPORTED = [  # (what, accuracy, macro F1, label). Old numbers, never new results.
    ("Thesis, CK+, 595 features, tuned. Random frame split, test set also used for selection",
     "91.95%", "0.90", "reported (thesis)"),
    ("Research report, student expression data, k-NN / logistic regression / linear SVM",
     "0.914 / 0.922 / 0.921", "0.250 / 0.272 / 0.243", "reported (paper)"),
]
EXPECTED = {  # file name -> description shown when the file does not exist yet
    "simulated_leakage_demo.json": "Evaluation leakage demo",
    "simulated_selection_leak.json": "Feature-selection leak demo",
    "simulated_imbalance_study.json": "Study 2: class imbalance",
    "study1_uibvfedplus_light.json": "Study 1: UIBVFEDPlus-Light",
    "study3_ferplus.json": "Study 3: FER2013 with FER+ labels",
    "study3_ood_uibvfedplus_light.json": "Study 3: out-of-distribution, UIBVFEDPlus-Light",
    "study3_ood_jaffe.json": "Study 3: out-of-distribution, JAFFE (you supply the data)",
    "private/study1_ckplus.json": "Study 1: CK+ (private data, numbers only)",
    "private/study1_kdef.json": "Study 1: KDEF (private data, numbers only)",
}


def fmt(row: dict, key: str) -> str:
    if row.get(key) is None:
        return "n/a"
    s = f"{row[key]:.3f}"
    sd = row.get(key + "_std")
    return f"{s} ± {sd:.3f}" if sd else s


def rows_from(path: Path) -> list[str]:
    rec = json.loads(path.read_text(encoding="utf-8"))
    table = rec.get("metrics", {}).get("table", [])
    out = []
    for r in table:
        out.append(f"| {rec['name']}: {r['name']} | {rec['dataset']} | {fmt(r, 'accuracy')} "
                   f"| {fmt(r, 'macro_f1')} | {rec['label']} |")
    return out


def build(results_dir: Path) -> str:
    lines = ["| Result | Data | Accuracy | Macro F1 | Label |", "|---|---|---|---|---|"]
    for what, acc, f1, label in REPORTED:
        lines.append(f"| {what} | thesis / report | {acc} | {f1} | {label} |")
    for fname, desc in EXPECTED.items():
        p = results_dir / fname
        if p.exists():
            lines += rows_from(p) or [f"| {desc} | n/a | no table in file | n/a | see file |"]
        else:
            lines.append(f"| {desc} | n/a | not measured | not measured | not measured |")
    for p in sorted(results_dir.glob("*.json")):
        if p.name not in EXPECTED:
            lines += rows_from(p)
    return "\n".join(lines)


def replace_block(text: str, table: str) -> str:
    if START not in text or END not in text:
        raise ValueError(f"markers {START} / {END} not found in README.md")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{table}\n{END}{tail}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.parse_args(argv)
    readme = ROOT / "README.md"
    old = readme.read_text(encoding="utf-8")
    new = replace_block(old, build(ROOT / "results"))
    if "--check" in argv:
        if new != old:
            print("README results table is out of date. Run: python scripts/make_results_table.py")
            return 1
        print("results table up to date")
        return 0
    readme.write_text(new, encoding="utf-8")
    print("results table written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
