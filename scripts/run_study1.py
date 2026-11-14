#!/usr/bin/env python3
"""Study 1: landmark features, Mutual Information selection and classifiers, evaluated with the
subject-wise protocol and, for comparison, the random and legacy protocols.

    python scripts/run_study1.py --features data/cache/uib.npz --dataset uibvfedplus_light \\
        --label "measured (local machine)"

Datasets the registry marks private_results are written to results/private/ (numbers only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fel.data import get_dataset  # noqa: E402
from fel.results import ALLOWED_LABELS, write_result  # noqa: E402
from fel.studies import load_features_npz, run_study1  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--dataset", required=True, help="registry name, for example ckplus")
    ap.add_argument("--label", required=True, choices=ALLOWED_LABELS)
    ap.add_argument("--models", default="majority,logreg,mlp",
                    help="comma list from majority,logreg,knn1,mlp,fernet")
    ap.add_argument("--feature-sets", default="",
                    help="comma list; default all four (see fel.studies.build_feature_sets)")
    ap.add_argument("--keep", type=float, default=0.75, help="Mutual Information keep fraction")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    entry = get_dataset(a.dataset)
    private = entry["role"] == "private_results"
    name = f"study1_{a.dataset}"
    folder = ROOT / "results" / ("private" if private else "")
    out = Path(a.out) if a.out else folder / f"{name}.json"
    data = load_features_npz(a.features)
    res = run_study1(
        data,
        feature_sets=[s for s in a.feature_sets.split(",") if s] or None,
        models=a.models.split(","), keep=a.keep, n_splits=a.splits, seed=a.seed,
    )
    protocol = {"split": f"subject-wise {a.splits}-fold, inner group-aware validation split",
                "selection": f"Mutual Information, keep {a.keep}, fitted inside training folds",
                "features": "pairwise distances of 40 selected landmarks, divided by inter-ocular"
                            " distance; see docs/study1.md"}
    write_result(out, name, a.label, a.dataset, protocol, res, seed=a.seed)
    print(f"wrote {out}")
    for r in res["table"]:
        print(f"{r['name'][:90]:90} acc {r['accuracy']:.3f}  macro F1 {r['macro_f1']:.3f}")
    print("Next: python scripts/make_results_table.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
