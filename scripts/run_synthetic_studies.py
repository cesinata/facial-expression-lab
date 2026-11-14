#!/usr/bin/env python3
"""Run the three simulated studies and write results/simulated_*.json (label: simulated).

Takes about a minute on a laptop. No data or network needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fel.results import write_result  # noqa: E402
from fel.studies import imbalance_demo, leakage_demo, selection_leak_demo  # noqa: E402


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = ROOT / "results"
    jobs = [
        ("simulated_leakage_demo.json", "leakage demo", leakage_demo,
         {"split": "legacy vs random 5-fold vs subject-wise 5-fold, 5 seeds"}),
        ("simulated_selection_leak.json", "selection leak demo", selection_leak_demo,
         {"split": "selection on all rows vs inside folds, 5 seeds, pure-noise features"}),
        ("simulated_imbalance_study.json", "class imbalance", imbalance_demo,
         {"split": "user-grouped 5-fold, 1 data seed"}),
    ]
    for fname, name, fn, protocol in jobs:
        print(f"running {name} ...", flush=True)
        write_result(out / fname, name, "simulated", "synthetic", protocol,
                     fn(), seed=0)
        print(f"  wrote results/{fname}")
    print("Next: python scripts/make_results_table.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
