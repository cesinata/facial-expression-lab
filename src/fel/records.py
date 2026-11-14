"""Turn dataset folders into (path, label, group) records.

Only filenames and folder names are read here. No images are opened. Group is the
subject (or actor, or character), used for subject-wise splitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

KDEF_EXPRESSIONS = {
    "AF": "fear", "AN": "angry", "DI": "disgust", "HA": "happy",
    "NE": "neutral", "SA": "sad", "SU": "surprise",
}
KDEF_PATTERN = re.compile(
    r"^(?P<session>[AB])(?P<gender>[FM])(?P<actor>\d{2})(?P<expr>AF|AN|DI|HA|NE|SA|SU)"
    r"(?P<angle>FL|HL|S|HR|FR)\.(jpg|jpeg)$",
    re.IGNORECASE,
)
CK_EMOTIONS = {0: "neutral", 1: "angry", 2: "contempt", 3: "disgust", 4: "fear",
               5: "happy", 6: "sad", 7: "surprise"}
CK_FRAME = re.compile(r"^(?P<subject>S\d{3})_(?P<seq>\d{3})_(?P<frame>\d{8})\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class Record:
    path: Path
    label: str
    group: str


def parse_kdef_name(name: str) -> dict[str, str] | None:
    m = KDEF_PATTERN.match(name)
    if not m:
        return None
    d = {k: v.upper() for k, v in m.groupdict().items()}
    d["label"] = KDEF_EXPRESSIONS[d["expr"]]
    d["group"] = f"{d['gender']}{d['actor']}"  # the same actor appears in sessions A and B
    return d


def from_kdef(root: str | Path, angles: tuple[str, ...] = ("S",)) -> list[Record]:
    """KDEF records for the chosen camera angles (default: straight-on only)."""
    out = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        d = parse_kdef_name(p.name)
        if d and d["angle"] in angles:
            out.append(Record(p, d["label"], d["group"]))
    return out


def from_regex(
    root: str | Path,
    label_regex: str,
    group_regex: str,
    label_map: dict[str, str] | None = None,
) -> list[Record]:
    """Records from any folder, using regexes on the path relative to root.

    Each regex needs one capture group. Example: label_regex=r"/(\\w+)_\\d+\\.png$".
    Files that do not match both regexes are skipped (and counted by the caller).
    """
    root = Path(root)
    lre, gre = re.compile(label_regex), re.compile(group_regex)
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = "/" + p.relative_to(root).as_posix()
        lm, gm = lre.search(rel), gre.search(rel)
        if lm and gm:
            lab = lm.group(1)
            lab = label_map.get(lab, lab) if label_map else lab
            out.append(Record(p, lab, gm.group(1)))
    return out


def read_ck_emotion(path: Path) -> str | None:
    """CK+ emotion files hold one number (for example 3.0000000e+00)."""
    try:
        code = int(float(path.read_text().strip().split()[0]))
    except (ValueError, IndexError, OSError):
        return None
    return CK_EMOTIONS.get(code)


def from_ckplus(
    images_dir: str | Path, emotion_dir: str | Path, frames: str = "thesis"
) -> list[Record]:
    """CK+ records from sequences that carry a provider emotion label.

    frames='thesis': first frame as neutral plus the last three frames as the emotion
                     (the frame choice used in the thesis code).
    frames='peak3':  last three frames as the emotion only.
    Sequences with no provider label are skipped, never labelled by hand.
    """
    if frames not in ("thesis", "peak3"):
        raise ValueError("frames must be 'thesis' or 'peak3'")
    images_dir, emotion_dir = Path(images_dir), Path(emotion_dir)
    out: list[Record] = []
    for efile in sorted(emotion_dir.rglob("*_emotion.txt")):
        label = read_ck_emotion(efile)
        if label is None:
            continue
        seq_rel = efile.parent.relative_to(emotion_dir)
        frames_list = sorted(
            f for f in (images_dir / seq_rel).glob("*.png") if CK_FRAME.match(f.name)
        )
        if len(frames_list) < 4:
            continue
        subject = seq_rel.parts[0]
        for f in frames_list[-3:]:
            out.append(Record(f, label, subject))
        if frames == "thesis":
            out.append(Record(frames_list[0], "neutral", subject))
    return out


JAFFE_EXPRESSIONS = {"AN": "anger", "DI": "disgust", "FE": "fear", "HA": "happiness",
                     "NE": "neutral", "SA": "sadness", "SU": "surprise"}


def from_jaffe(root: str | Path) -> list[Record]:
    """JAFFE records from names like KA.AN1.39.tiff (subject, expression code + number, id).

    Labels use FER+ class names. Check a few names with scripts/probe_dataset.py first.
    """
    out = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = p.name.split(".")
        if len(parts) >= 3 and parts[1][:2].upper() in JAFFE_EXPRESSIONS:
            out.append(Record(p, JAFFE_EXPRESSIONS[parts[1][:2].upper()], parts[0].upper()))
    return out
