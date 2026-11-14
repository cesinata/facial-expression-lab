"""FER2013 and FER+ loading and label rules (numpy and pandas only).

FER2013: fer2013.csv with columns emotion, pixels (space-separated 48x48), Usage.
FER+:    fer2013new.csv with the same row order, plus vote counts per emotion.
The column names are checked on load. If they differ, a clear error is raised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FERPLUS_CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear",
                   "contempt"]
FERPLUS_EXTRA = ["unknown", "NF"]  # "NF" means not a face
SHARED_WITH_SIX = {  # FER+ class -> the six-emotion names used by UIBVFEDPlus-Light
    "anger": "anger", "disgust": "disgust", "fear": "fear",
    "happiness": "joy", "sadness": "sadness", "surprise": "surprise",
}


def _read(path: str) -> pd.DataFrame:
    """Read a CSV and strip spaces from the column names (icml_face_data.csv has ' pixels')."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_fer2013_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (pixels uint8 (n, 48, 48), usage array). Works for fer2013.csv and
    icml_face_data.csv (same rows, same order)."""
    df = _read(path)
    for col in ("pixels", "Usage"):
        if col not in df.columns:
            raise ValueError(f"fer2013.csv: missing column {col!r}; found {list(df.columns)}")
    arr = np.stack([np.fromstring(s, dtype=np.uint8, sep=" ") for s in df["pixels"]])
    if arr.shape[1] != 48 * 48:
        raise ValueError(f"expected 2304 pixels per row, got {arr.shape[1]}")
    return arr.reshape(-1, 48, 48), df["Usage"].astype(str).str.strip().to_numpy()


def load_ferplus_votes(path: str) -> np.ndarray:
    """Return vote counts (n, 10) in the order FERPLUS_CLASSES + FERPLUS_EXTRA."""
    df = _read(path)
    need = FERPLUS_CLASSES + FERPLUS_EXTRA
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"fer2013new.csv: missing columns {missing}; found {list(df.columns)}")
    return df[need].fillna(0).to_numpy(dtype=int)


def load_ferplus_usage(path: str) -> np.ndarray | None:
    """The Usage column of fer2013new.csv, or None if the file has no such column."""
    df = _read(path)
    if "Usage" not in df.columns:
        return None
    return df["Usage"].astype(str).str.strip().to_numpy()


def usage_mismatch_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Share of rows where two Usage arrays disagree. Used to check that FER+ rows line up
    with FER2013 rows. Raises if the lengths differ."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        raise ValueError(f"row counts differ: {len(a)} vs {len(b)}")
    return float((a != b).mean())


def ferplus_majority(votes: np.ndarray) -> np.ndarray:
    """Majority-vote label per row, or -1 when the row should be dropped.

    A row is dropped when 'unknown' plus 'NF' votes outnumber the top emotion, or when the
    top emotion is tied with another emotion.
    """
    votes = np.asarray(votes)
    emo = votes[:, : len(FERPLUS_CLASSES)]
    extra = votes[:, len(FERPLUS_CLASSES):].sum(axis=1)
    top = emo.max(axis=1)
    label = emo.argmax(axis=1)
    tied = (emo == top[:, None]).sum(axis=1) > 1
    drop = (extra > top) | tied | (top == 0)
    label = label.astype(int)
    label[drop] = -1
    return label


def split_by_usage(usage: np.ndarray) -> dict[str, np.ndarray]:
    return {u: np.flatnonzero(usage == u) for u in ("Training", "PublicTest", "PrivateTest")}
