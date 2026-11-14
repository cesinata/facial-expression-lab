"""Single-sample latency measurement. Report where it was measured (host CPU, Colab T4, browser)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


def measure_latency(
    fn: Callable[[Any], Any],
    sample: Any,
    n_runs: int = 200,
    n_warmup: int = 20,
    sync: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Time fn(sample) one call at a time. Returns median and 95th percentile in milliseconds.

    Pass `sync` (for example torch.cuda.synchronize) when timing GPU code, otherwise the
    timer stops before the GPU has finished.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    for _ in range(n_warmup):
        fn(sample)
    times = np.empty(n_runs)
    for i in range(n_runs):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn(sample)
        if sync:
            sync()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "n_runs": int(n_runs),
    }
