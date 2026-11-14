#!/usr/bin/env python3
"""Study 3: train a small image model on FER2013 with FER+ labels, then evaluate it once, export
it to ONNX and test it out of distribution. Needs PyTorch and timm (use a Colab GPU for `train`).

    python scripts/study3.py train    --fer-csv fer2013.csv --ferplus-csv fer2013new.csv
    python scripts/study3.py evaluate --fer-csv fer2013.csv --ferplus-csv fer2013new.csv \\
                                      --label "measured (Colab T4)"
    python scripts/study3.py ood --npz data/cache/uib.npz --dataset uibvfedplus_light \\
                                 --label "measured (local machine)"

`evaluate` reads the PrivateTest split once and refuses to run twice (use --allow-repeat to
override, and say so in the AI log). Everything except the model files is small JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fel import fer  # noqa: E402
from fel.calibration import fit_temperature, softmax  # noqa: E402
from fel.latency import measure_latency  # noqa: E402
from fel.metrics import classification_metrics  # noqa: E402
from fel.results import ALLOWED_LABELS, write_result  # noqa: E402

CLASSES = fer.FERPLUS_CLASSES


def load_split(fer_csv: str, ferplus_csv: str):
    pixels, usage = fer.load_fer2013_csv(fer_csv)
    labels = fer.ferplus_majority(fer.load_ferplus_votes(ferplus_csv))
    if len(labels) != len(pixels):
        raise SystemExit(f"row counts differ: fer2013 {len(pixels)} vs fer2013new {len(labels)}")
    plus_usage = fer.load_ferplus_usage(ferplus_csv)
    if plus_usage is not None:
        bad = fer.usage_mismatch_fraction(usage, plus_usage)
        if bad > 0.01:
            raise SystemExit(f"FER2013 and FER+ rows do not line up ({bad:.1%} of Usage values "
                             "differ). Check that both files are the originals, same order.")
        print(f"FER2013 and FER+ Usage columns agree on {1 - bad:.2%} of rows")
    keep = labels >= 0
    print(f"rows kept after FER+ rules: {keep.sum()}/{len(keep)}")
    parts = fer.split_by_usage(usage)
    return {k: (pixels[idx[keep[idx]]], labels[idx[keep[idx]]]) for k, idx in parts.items()}


def cmd_train(a) -> int:
    import torch

    from fel import train_fer as T

    d = load_split(a.fer_csv, a.ferplus_csv)
    model = T.build_model(a.arch, len(CLASSES), pretrained=not a.no_pretrained)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info = T.train_classifier(model, *d["Training"], *d["PublicTest"], n_classes=len(CLASSES),
                              epochs=a.epochs, seed=a.seed)
    torch.save(model.state_dict(), out / "model.pt")
    meta = {"arch": a.arch, "pretrained": not a.no_pretrained, "epochs": a.epochs,
            "seed": a.seed, "train_seconds": time.time() - t0, **info}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"best epoch {info['best_epoch']} (validation macro F1 {info['best_val_macro_f1']:.3f})")
    return 0


def _load_trained(out: Path):
    import torch

    from fel import train_fer as T

    meta = json.loads((out / "meta.json").read_text())
    model = T.build_model(meta["arch"], len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu"))
    return model.eval(), meta


def cmd_evaluate(a) -> int:
    from fel import train_fer as T

    out = Path(a.out_dir)
    marker = out / "test_evaluated.txt"
    if marker.exists() and not a.allow_repeat:
        raise SystemExit("PrivateTest was already evaluated once. Use --allow-repeat to override.")
    d = load_split(a.fer_csv, a.ferplus_csv)
    model, meta = _load_trained(out)
    xv, yv = d["PublicTest"]
    xt, yt = d["PrivateTest"]
    temp = fit_temperature(T.predict_logits(model, xv, device="cpu"), yv)
    logits = T.predict_logits(model, xt, device="cpu")
    marker.write_text("PrivateTest evaluated once.\n")
    idx = list(range(len(CLASSES)))
    base = classification_metrics(yt, logits.argmax(1), idx, softmax(logits))
    cal = classification_metrics(yt, logits.argmax(1), idx, softmax(logits, temp))
    rows = [{"name": "PyTorch FP32", "accuracy": base["accuracy"], "macro_f1": base["macro_f1"]}]
    extra = {"temperature": temp, "ece_before": base["ece"], "ece_after": cal["ece"],
             "per_class_recall": {CLASSES[i]: base["per_class"][str(i)]["recall"] for i in idx},
             "never_predicted": [CLASSES[int(i)] for i in base["never_predicted"]],
             "best_epoch": meta["best_epoch"], "n_test": int(len(yt))}
    onnx_path = T.export_onnx(model, out / "model.onnx")
    int8 = T.quantize_int8(onnx_path, out / "model_int8.onnx")
    extra["size_mb"] = {"onnx_fp32": onnx_path.stat().st_size / 1e6}
    sample = xt[:1]
    for tag, path in (("ONNX FP32", onnx_path), ("ONNX INT8", int8)):
        if path is None:
            continue
        sess = T.onnx_session(path)
        lg = T.onnx_logits(sess, xt)
        m = classification_metrics(yt, lg.argmax(1), idx)
        rows.append({"name": tag, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        if tag == "ONNX FP32":
            extra["max_abs_logit_diff_pytorch_vs_onnx"] = float(np.abs(lg - logits).max())
        else:
            extra["size_mb"]["onnx_int8"] = path.stat().st_size / 1e6
        extra[f"latency_{tag.replace(' ', '_').lower()}"] = measure_latency(
            lambda x, sess=sess: T.onnx_logits(sess, x), sample)
    write_result(ROOT / "results" / "study3_ferplus.json", "study3", a.label, "ferplus",
                 {"split": "FER2013 Usage split; PrivateTest read once; no subject IDs, so "
                           "results are optimistic",
                  "model": meta["arch"], "labels": "FER+ majority vote, unknown/NF dropped"},
                 {"table": rows, "extra": extra}, seed=meta["seed"])
    print(json.dumps(rows, indent=1))
    print("Latency was measured on THIS machine. Pass --label to say which.")
    return 0


def cmd_ood(a) -> int:
    from fel import train_fer as T

    out = Path(a.out_dir)
    model, _ = _load_trained(out)
    data = np.load(a.npz, allow_pickle=False)
    if "crops" not in data.files:
        raise SystemExit("this .npz has no crops. Re-run extract_features.py with --crops")
    names = [fer.SHARED_WITH_SIX.get(s, s) for s in data["labels"].tolist()]
    names = [{"joy": "happiness"}.get(n, n) for n in names]
    present = [c for c in CLASSES if c in set(names)]
    cols = [CLASSES.index(c) for c in present]
    mask = np.array([n in present for n in names])
    logits = T.predict_logits(model, data["crops"][mask], device="cpu")[:, cols]
    y = np.array([present.index(n) for n, m in zip(names, mask, strict=True) if m])
    m = classification_metrics(y, logits.argmax(1), list(range(len(present))))
    row = {"name": f"restricted to {len(present)} shared classes", "accuracy": m["accuracy"],
           "macro_f1": m["macro_f1"]}
    extra = {"classes": present, "n": int(mask.sum()),
             "per_class_recall": {c: m["per_class"][str(i)]["recall"]
                                  for i, c in enumerate(present)}}
    write_result(ROOT / "results" / f"study3_ood_{a.dataset}.json", "study3 out-of-distribution",
                 a.label, a.dataset, {"split": "no training on this dataset; argmax restricted to "
                                              "the classes it shares with FER+"},
                 {"table": [row], "extra": extra}, seed=0)
    print(json.dumps(row, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "evaluate", "ood"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default="data/cache/study3")
        if name != "ood":
            p.add_argument("--fer-csv", required=True)
            p.add_argument("--ferplus-csv", required=True)
        if name == "train":
            p.add_argument("--arch", default="mobilenetv3_small_100")
            p.add_argument("--epochs", type=int, default=15)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--no-pretrained", action="store_true")
        else:
            p.add_argument("--label", required=True, choices=ALLOWED_LABELS)
        if name == "evaluate":
            p.add_argument("--allow-repeat", action="store_true")
        if name == "ood":
            p.add_argument("--npz", required=True)
            p.add_argument("--dataset", required=True)
    a = ap.parse_args()
    return {"train": cmd_train, "evaluate": cmd_evaluate, "ood": cmd_ood}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
