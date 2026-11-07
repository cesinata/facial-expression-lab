"""Dataset registry and the download guard.

The registry (datasets.yaml) is the single source of truth for dataset terms.
`download` prints the licence and refuses anything the registry does not allow
the code to fetch. Dataset-specific fetchers are added in a later phase; until
then an allowed dataset raises NotImplementedError after the checks pass.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FIELDS = (
    "name",
    "title",
    "what",
    "access",
    "licence",
    "licence_status",
    "public",
    "fetch",
    "role",
    "status",
    "last_checked",
)
LICENCE_STATUSES = ("confirmed", "partly_confirmed", "unconfirmed")
FETCH_MODES = ("runtime", "user_supplied", "never")
ROLES = ("used", "private_results", "not_used")
STATUSES = ("planned", "done")


def default_registry_path() -> Path:
    """Registry location: FEL_DATASETS_YAML if set, else datasets.yaml at the repo root."""
    env = os.environ.get("FEL_DATASETS_YAML")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "datasets.yaml"


def load_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the registry as {name: entry}. Raises ValueError on duplicate names."""
    p = Path(path) if path else default_registry_path()
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    entries = raw.get("datasets", [])
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        name = e.get("name")
        if name in out:
            raise ValueError(f"duplicate dataset name in registry: {name}")
        out[name] = e
    return out


def validate_registry(registry: dict[str, dict[str, Any]]) -> list[str]:
    """Return a list of problems. An empty list means the registry is consistent."""
    problems: list[str] = []
    for name, e in registry.items():
        for field in REQUIRED_FIELDS:
            if field not in e:
                problems.append(f"{name}: missing field '{field}'")
        if e.get("licence_status") not in LICENCE_STATUSES:
            problems.append(f"{name}: bad licence_status {e.get('licence_status')!r}")
        if e.get("fetch") not in FETCH_MODES:
            problems.append(f"{name}: bad fetch {e.get('fetch')!r}")
        if e.get("role") not in ROLES:
            problems.append(f"{name}: bad role {e.get('role')!r}")
        if e.get("status") not in STATUSES:
            problems.append(f"{name}: bad status {e.get('status')!r}")
        if not isinstance(e.get("public"), bool):
            problems.append(f"{name}: 'public' must be true or false")
        if e.get("public") is False and e.get("fetch") != "never":
            problems.append(f"{name}: public is false, so fetch must be 'never'")
        if e.get("role") in ("private_results", "not_used") and e.get("fetch") != "never":
            problems.append(f"{name}: role {e.get('role')} requires fetch 'never'")
        lc = e.get("last_checked")
        if lc is not None and not isinstance(lc, (_dt.date, str)):
            problems.append(f"{name}: last_checked must be a date or null")
        if e.get("licence_status") == "confirmed" and lc is None:
            problems.append(f"{name}: licence_status 'confirmed' needs a last_checked date")
    return problems


def get_dataset(name: str, registry: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    reg = registry if registry is not None else load_registry()
    if name not in reg:
        raise KeyError(f"dataset '{name}' is not in datasets.yaml")
    return reg[name]


def licence_notice(name: str, registry: dict[str, dict[str, Any]] | None = None) -> str:
    """Text printed before any download."""
    e = get_dataset(name, registry)
    checked = e.get("last_checked") or "not checked"
    lines = [
        f"Dataset: {e['title']} ({name})",
        f"Access: {e['access']}",
        f"Licence: {e['licence']} [{e['licence_status']}, last checked: {checked}]",
    ]
    if e.get("source_url"):
        lines.append(f"Source: {e['source_url']}")
    if e.get("notes"):
        lines.append(f"Notes: {e['notes']}")
    return "\n".join(lines)


def download(
    name: str,
    dest: str | Path = "data/raw",
    registry: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Print the licence, then fetch the dataset if the registry allows it.

    Refuses datasets marked public: false, fetch: never or fetch: user_supplied.
    """
    reg = registry if registry is not None else load_registry()
    e = get_dataset(name, reg)
    print(licence_notice(name, reg))
    if e["public"] is False or e["fetch"] == "never":
        raise PermissionError(
            f"'{name}' is not public or must not be downloaded by this code. "
            "Obtain it from the provider and pass its folder path to the script."
        )
    if e["fetch"] == "user_supplied":
        raise PermissionError(
            f"'{name}' must be downloaded by you after reading the provider's terms. "
            "Pass the folder path to the notebook or script instead."
        )
    if e["licence_status"] == "unconfirmed":
        print(
            "WARNING: the licence for this dataset is unconfirmed in datasets.yaml. "
            "Read the provider's terms before continuing."
        )
    raise NotImplementedError(
        f"The fetcher for '{name}' is added in a later phase (destination: {Path(dest)})."
    )
