#!/usr/bin/env python3
"""Run MediaPipe Face Landmarker over a dataset and save landmarks, blendshapes and labels.

Output is a local .npz in data/cache/ (ignored by git). It holds numbers only: no image
paths, no images unless --crops is used (48x48 grey face crops for the image model).

Examples
    python scripts/extract_features.py --source kdef --root ~/data/KDEF --out data/cache/kdef.npz
    python scripts/extract_features.py --source ckplus --images-dir ~/data/CK/cohn-kanade-images \\
        --emotion-dir ~/data/CK/Emotion --out data/cache/ckplus.npz
    python scripts/extract_features.py --source regex --root ~/data/UIB \\
        --label-regex '/(\\w+)_light' --group-regex '^/(\\w+)/' --out data/cache/uib.npz --crops
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fel import landmarks as L  # noqa: E402
from fel import records as R  # noqa: E402


def parse_label_map(s: str | None) -> dict[str, str] | None:
    if not s:
        return None
    return dict(kv.split("=", 1) for kv in s.split(","))


def get_records(a) -> list:
    if a.source == "kdef":
        return R.from_kdef(a.root, tuple(a.angles.split(",")))
    if a.source == "jaffe":
        return R.from_jaffe(a.root)
    if a.source == "ckplus":
        return R.from_ckplus(a.images_dir, a.emotion_dir, a.frames)
    if not (a.label_regex and a.group_regex):
        raise SystemExit("--source regex needs --label-regex and --group-regex")
    return R.from_regex(a.root, a.label_regex, a.group_regex, parse_label_map(a.label_map))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", required=True, choices=["kdef", "ckplus", "jaffe", "regex"])
    ap.add_argument("--root")
    ap.add_argument("--images-dir")
    ap.add_argument("--emotion-dir")
    ap.add_argument("--angles", default="S", help="KDEF angles, comma separated (default S)")
    ap.add_argument("--frames", default="thesis", choices=["thesis", "peak3"])
    ap.add_argument("--label-regex")
    ap.add_argument("--group-regex")
    ap.add_argument("--label-map", help="for example joy=happiness,anger=anger")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crops", action="store_true", help="also save 48x48 grey face crops")
    ap.add_argument("--limit", type=int, default=0,
                    help="N records evenly spaced across all records (debugging)")
    ap.add_argument("--model-dir", default="data/cache")
    a = ap.parse_args()

    recs = get_records(a)
    if a.limit and a.limit < len(recs):
        keep = np.unique(np.linspace(0, len(recs) - 1, a.limit).round().astype(int))
        recs = [recs[i] for i in keep]
    if not recs:
        raise SystemExit("no records found. Run scripts/probe_dataset.py first.")
    print(f"{len(recs)} records, {len({r.group for r in recs})} subjects")
    print("labels:", dict(collections.Counter(r.label for r in recs)))

    extractor = L.FaceExtractor(L.ensure_model(a.model_dir))
    ok, lms, bss, boxes, failed = [], [], [], [], []
    for i, r in enumerate(recs, 1):
        res = extractor.extract(r.path)
        ok.append(res is not None)
        if res is None:
            failed.append(r.path.name)
        else:
            lms.append(res[0])
            bss.append(res[1])
            boxes.append(res[2])
        if i % 200 == 0:
            print(f"  {i}/{len(recs)}", flush=True)
    ok = np.array(ok)
    print(f"face found in {ok.sum()}/{len(ok)} ({100 * ok.mean():.1f}%)")
    if failed:
        print("first files with no face:", failed[:10])
    kept = [r for r, k in zip(recs, ok, strict=True) if k]
    out = {
        "landmarks": np.stack(lms).astype(np.float32),
        "blendshapes": np.stack(bss).astype(np.float32),
        "boxes": np.array(boxes, dtype=np.int32),
        "labels": np.array([r.label for r in kept]),
        "groups": np.array([r.group for r in kept]),
    }
    if a.crops:
        from fel.imageio import load_gray_crop

        out["crops"] = np.stack([load_gray_crop(r.path, tuple(b)) for r, b in
                                 zip(kept, boxes, strict=True)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print("kept labels:", dict(collections.Counter(out["labels"].tolist())))
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
