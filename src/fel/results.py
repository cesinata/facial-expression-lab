"""Write metrics to results/*.json with a mandatory provenance label.

Every number in this repo is labelled. A result file without a valid label is
refused, and so is one that names a dataset missing from datasets.yaml.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from pathlib import Path
from typing import Any

from fel.data import load_registry

ALLOWED_LABELS = (
    "measured (Colab T4)",
    "measured (host CPU)",
    "measured (local machine)",
    "measured (browser)",
    "estimated",
    "simulated",
    "reported (thesis)",
    "reported (paper)",
    "not measured",
)


def library_versions() -> dict[str, str]:
    """Versions of the libraries that affect results. Missing libraries are skipped."""
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "torch", "onnxruntime", "mediapipe", "timm"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "unknown"))
        except ImportError:
            continue
    return out


def write_result(
    path: str | Path,
    name: str,
    label: str,
    dataset: str,
    protocol: dict[str, Any],
    metrics: dict[str, Any],
    seed: int,
    registry: dict[str, dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one result file. Returns the path."""
    if label not in ALLOWED_LABELS:
        raise ValueError(f"label {label!r} not allowed; use one of {ALLOWED_LABELS}")
    reg = registry if registry is not None else load_registry()
    if dataset not in reg and dataset != "synthetic":
        raise ValueError(f"dataset {dataset!r} is not in datasets.yaml (or 'synthetic')")
    record = {
        "name": name,
        "label": label,
        "dataset": dataset,
        "protocol": protocol,
        "seed": seed,
        "metrics": metrics,
        "versions": library_versions(),
        "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        record["extra"] = extra
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    return p
