#!/usr/bin/env python3
"""Record a browser latency measurement of the Study 3 ONNX model as results/*.json.

The number comes from the demo page in a real browser, never from this machine. Either:
  - press "Measure model time" on the demo page (200 timed runs after 20 warm-up runs), which
    prints median and 95th percentile, or
  - read the live "Median model time over the last 30 frames" line on the page.

    python scripts/record_browser_latency.py --median-ms 3.2 --n-runs 30 \\
        --method "live loop, median of the last 30 frames" \\
        --browser "Chrome (version not recorded)" --device "MacBook (macOS)" --date 2026-11-15

The label is always "measured (browser)". Browser, device and date are required.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fel.results import write_result  # noqa: E402

LABEL = "measured (browser)"


def build_record(median_ms: float, n_runs: int, browser: str, device: str, date: str,
                 method: str, p95_ms: float | None = None, ort_web: str = "") -> dict:
    """Return the protocol and metrics for write_result. Raises ValueError on bad input."""
    if median_ms <= 0:
        raise ValueError("median_ms must be positive")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if p95_ms is not None and p95_ms < median_ms:
        raise ValueError("p95_ms cannot be below median_ms")
    for field, value in (("browser", browser), ("device", device), ("method", method)):
        if not value.strip():
            raise ValueError(f"{field} is required")
    _dt.date.fromisoformat(date)  # raises ValueError if the date is not YYYY-MM-DD
    latency = {"median_ms": median_ms, "p95_ms": p95_ms, "n_runs": n_runs, "browser": browser,
               "device": device, "date": date, "method": method,
               "onnxruntime_web": ort_web or "not recorded"}
    row = (f"ONNX FP32 in the browser, median {median_ms:g} ms over {n_runs} runs, "
           f"{browser}, {device}, {date}")
    return {
        "protocol": {"model": "model.onnx (FP32) from release v1.0.0",
                     "input": "48x48 grey crop from the live webcam via the demo page",
                     "timing": "performance.now() around session.run only, one call at a time",
                     "method": method},
        "metrics": {"table": [{"name": row}], "extra": {"latency_browser": latency}},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--median-ms", type=float, required=True)
    ap.add_argument("--p95-ms", type=float, default=None)
    ap.add_argument("--n-runs", type=int, required=True)
    ap.add_argument("--browser", required=True, help='for example "Chrome 142"')
    ap.add_argument("--device", required=True, help='for example "MacBook (macOS)"')
    ap.add_argument("--date", required=True, help="YYYY-MM-DD, the day it was measured")
    ap.add_argument("--method", required=True)
    ap.add_argument("--ort-web", default="", help="onnxruntime-web version the page loaded")
    ap.add_argument("--out", default=str(ROOT / "results" / "study3_browser_latency.json"))
    a = ap.parse_args(argv)
    rec = build_record(a.median_ms, a.n_runs, a.browser, a.device, a.date, a.method,
                       a.p95_ms, a.ort_web)
    path = write_result(a.out, name="study3_browser_latency", label=LABEL, dataset="ferplus",
                        protocol=rec["protocol"], metrics=rec["metrics"], seed=0)
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
