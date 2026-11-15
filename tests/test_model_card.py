"""The model card must show the numbers in the result files, formatted the way it prints them."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CARD = (ROOT / "docs" / "model_card.md").read_text(encoding="utf-8")
S3 = ROOT / "results" / "study3_ferplus.json"
OOD = ROOT / "results" / "study3_ood_jaffe.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


@pytest.mark.skipif(not S3.exists(), reason="results/study3_ferplus.json not committed yet")
def test_model_card_matches_study3_results():
    m = _load(S3)
    rows = {r["name"]: r for r in m["table"]}
    extra = m["extra"]
    wanted = [f"{extra['n_test']:,}", f"{extra['ece_before']:.3f}", f"{extra['ece_after']:.3f}",
              f"{extra['temperature']:.3f}", f"{extra['max_abs_logit_diff_pytorch_vs_onnx']:.1e}",
              f"{extra['size_mb']['onnx_fp32']:.2f} MB", f"{extra['size_mb']['onnx_int8']:.2f} MB",
              f"best epoch {extra['best_epoch']}"]
    for name in ("PyTorch FP32", "ONNX FP32", "ONNX INT8"):
        wanted += [f"{rows[name]['accuracy']:.3f}", f"{rows[name]['macro_f1']:.3f}"]
    for tag in ("onnx_fp32", "onnx_int8"):
        lat = extra[f"latency_{tag}"]
        wanted += [f"{lat['median_ms']:.2f} ms", f"{lat['p95_ms']:.2f} ms"]
    for cls, recall in extra["per_class_recall"].items():
        wanted.append(f"| {cls} | {recall:.3f} |")
    assert extra["never_predicted"] == []
    missing = [w for w in wanted if w not in CARD]
    assert not missing, f"model card is missing or differs on: {missing}"


@pytest.mark.skipif(not OOD.exists(), reason="results/study3_ood_jaffe.json not committed yet")
def test_model_card_matches_jaffe_ood_result():
    row = _load(OOD)["table"][0]
    line = f"| {row['accuracy']:.3f} | {row['macro_f1']:.3f} |"
    assert line in CARD


@pytest.mark.skipif(not S3.exists(), reason="results/study3_ferplus.json not committed yet")
def test_release_notes_match_study3_results():
    notes = (ROOT / "docs" / "release_notes.md").read_text(encoding="utf-8")
    m = _load(S3)
    rows = {r["name"]: r for r in m["table"]}
    extra = m["extra"]
    wanted = [f"{extra['n_test']:,}", f"{extra['size_mb']['onnx_fp32']:.2f} MB",
              f"{round(extra['size_mb']['onnx_fp32'] * 1e6):,} bytes",
              f"{extra['size_mb']['onnx_int8']:.2f} MB", f"best epoch {extra['best_epoch']}",
              f"{extra['ece_before']:.3f}", f"{extra['ece_after']:.3f}",
              f"{extra['temperature']:.3f}", f"{extra['max_abs_logit_diff_pytorch_vs_onnx']:.1e}",
              f"{extra['latency_onnx_fp32']['median_ms']:.2f} ms",
              f"{extra['latency_onnx_fp32']['p95_ms']:.2f} ms",
              f"{extra['latency_onnx_int8']['median_ms']:.2f} ms"]
    for name in ("ONNX FP32", "ONNX INT8"):
        wanted += [f"{rows[name]['accuracy']:.3f}", f"{rows[name]['macro_f1']:.3f}"]
    for cls, recall in extra["per_class_recall"].items():
        wanted.append(f"{cls} {recall:.3f}")
    missing = [w for w in wanted if w not in notes]
    assert not missing, f"release notes are missing or differ on: {missing}"


@pytest.mark.skipif(not S3.exists(), reason="results/study3_ferplus.json not committed yet")
def test_demo_names_the_three_weakest_classes_with_their_recall():
    recall = _load(S3)["extra"]["per_class_recall"]
    weakest = sorted(recall, key=recall.get)[:3]
    for doc in ("demo/index.html", "demo/README.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        for cls in weakest:
            assert f"{cls} ({recall[cls]:.3f})" in text or f"{cls} {recall[cls]:.3f}" in text, (
                f"{doc} does not show {cls} with recall {recall[cls]:.3f}")


BROWSER = ROOT / "results" / "study3_browser_latency.json"


def test_model_card_browser_latency_row_matches_the_result_file():
    row = next(line for line in CARD.splitlines() if line.startswith("| Browser latency |"))
    if not BROWSER.exists():
        assert "not measured" in row
        return
    rec = json.loads(BROWSER.read_text(encoding="utf-8"))
    assert rec["label"] == "measured (browser)"
    lat = rec["metrics"]["extra"]["latency_browser"]
    for wanted in (f"median {lat['median_ms']:g} ms", f"over {lat['n_runs']} runs",
                   "measured (browser)", lat["browser"], lat["device"], lat["date"]):
        assert wanted in row, f"browser latency row is missing {wanted!r}"
    if lat["p95_ms"] is None:
        assert "95th percentile not recorded" in row
    else:
        assert f"{lat['p95_ms']:g} ms" in row
