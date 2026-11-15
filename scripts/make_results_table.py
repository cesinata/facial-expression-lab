#!/usr/bin/env python3
"""Fill the results tables from results/*.json (and results/private/*.json).

    python scripts/make_results_table.py           # rewrite both tables
    python scripts/make_results_table.py --check   # exit 1 if either is out of date (CI)

docs/results.md gets the full table. README.md gets a short headline table: a fixed list of rows
picked by name from the same files, plus one line naming the results that are not measured.
Nothing is typed by hand: a number appears only if a result file holds it. Expected results
that have no file yet show "not measured". Reported numbers from older work keep their label.
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
    "study3_browser_latency.json": "Study 3: browser latency",
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


def notes_from(results_dir: Path) -> list[str]:
    """Plain-language notes, written only when the result files show the condition."""
    notes = []
    p = results_dir / "study3_ferplus.json"
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        rows = {r["name"]: r for r in rec.get("metrics", {}).get("table", [])}
        notes.append("Study 3 uses FER2013, which has no subject IDs, so the same person can be "
                     "in several splits and its numbers are optimistic.")
        fp32, int8 = rows.get("ONNX FP32"), rows.get("ONNX INT8")
        if fp32 and int8 and int8["accuracy"] < 0.5 * fp32["accuracy"]:
            notes.append("The ONNX INT8 row is a measured failure: dynamic quantisation damaged "
                         "the model, so it is not published and the demo uses the FP32 file.")
    p = results_dir / "study3_browser_latency.json"
    if p.exists():
        lat = json.loads(p.read_text(encoding="utf-8"))["metrics"]["extra"]["latency_browser"]
        notes.append(f"The browser latency row has no accuracy: it is the model time in the demo "
                     f"page, median {lat['median_ms']:g} ms over {lat['n_runs']} runs, "
                     f"{lat['browser']}, {lat['device']}, {lat['date']} (measured (browser)).")
    return notes


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
    notes = notes_from(results_dir)
    if notes:
        lines += [""] + [f"Note: {n}" for n in notes]
    return "\n".join(lines)


# Headline rows for README.md: (file, row name in the file, text shown). The text names the row;
# every number comes from the file. A file that does not exist simply drops its rows.
HEADLINE = [
    ("study1_jaffe.json", "distances 2D | logreg | random 10-fold (rows)",
     "Study 1, JAFFE, logistic regression on 2D distances: random split of rows"),
    ("study1_jaffe.json", "distances 2D | logreg | legacy thesis protocol",
     "Study 1, same features: legacy thesis protocol"),
    ("study1_jaffe.json", "distances 2D | logreg | subject-wise 10-fold",
     "Study 1, same features: split by subject (leave-one-subject-out)"),
    ("simulated_imbalance_study.json", "logreg / baseline",
     "Study 2, one class near 92%: logistic regression, no fix"),
    ("simulated_imbalance_study.json", "linear_svc / class_weight",
     "Study 2: linear SVM with class weights"),
    ("study3_ferplus.json", "ONNX FP32",
     "Study 3, FER2013 PrivateTest: published ONNX FP32 model (optimistic: no subject IDs)"),
    ("study3_ferplus.json", "ONNX INT8",
     "Study 3: ONNX INT8 (measured failure, not published)"),
    ("study3_ood_jaffe.json", "restricted to 7 shared classes",
     "Study 3 model on JAFFE, not seen in training (7 shared classes)"),
]


def headline_rows(results_dir: Path) -> list[str]:
    cache: dict[str, dict] = {}
    out = []
    for fname, row_name, text in HEADLINE:
        p = results_dir / fname
        if not p.exists():
            continue
        if fname not in cache:
            cache[fname] = json.loads(p.read_text(encoding="utf-8"))
        rec = cache[fname]
        rows = {r["name"]: r for r in rec.get("metrics", {}).get("table", [])}
        if row_name not in rows:
            raise KeyError(f"{fname} has no row named {row_name!r}; update HEADLINE")
        r = rows[row_name]
        out.append(f"| {text} | {fmt(r, 'accuracy')} | {fmt(r, 'macro_f1')} | {rec['label']} |")
    return out


def build_headline(results_dir: Path) -> str:
    what, acc, f1, label = REPORTED[0]
    lines = ["| Result | Accuracy | Macro F1 | Label |", "|---|---|---|---|",
             f"| Thesis, CK+, random frame split, test set also used for selection | {acc} "
             f"| {f1} | {label} |"]
    lines += headline_rows(results_dir)
    lat = results_dir / "study3_browser_latency.json"
    extra_lines = []
    if lat.exists():
        b = json.loads(lat.read_text(encoding="utf-8"))["metrics"]["extra"]["latency_browser"]
        extra_lines.append(f"Browser latency of the published model: median {b['median_ms']:g} ms "
                           f"over {b['n_runs']} runs, {b['browser']}, {b['device']}, {b['date']} "
                           f"(measured (browser)).")
    missing = [desc for fname, desc in EXPECTED.items() if not (results_dir / fname).exists()]
    if missing:
        extra_lines.append("Not measured yet: " + "; ".join(missing) + ".")
    if extra_lines:
        lines += [""] + extra_lines
    return "\n".join(lines)


TARGETS = {"docs/results.md": build, "README.md": build_headline}


def replace_block(text: str, table: str) -> str:
    if START not in text or END not in text:
        raise ValueError(f"markers {START} / {END} not found")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{table}\n{END}{tail}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.parse_args(argv)
    stale = []
    for rel, builder in TARGETS.items():
        path = ROOT / rel
        old = path.read_text(encoding="utf-8")
        new = replace_block(old, builder(ROOT / "results"))
        if new != old:
            stale.append(rel)
            if "--check" not in argv:
                path.write_text(new, encoding="utf-8")
    if "--check" in argv:
        if stale:
            print("results tables out of date: " + ", ".join(stale)
                  + ". Run: python scripts/make_results_table.py")
            return 1
        print("results table up to date")
        return 0
    print("results tables written" if stale else "results tables already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
