"""MediaPipe Face Landmarker wrapper and landmark-derived features.

MediaPipe is imported only when FaceExtractor is created, so the rest of the package
and the tests work without it.

Region definitions are NOT the thesis region lists (the original pair lists were not
available). They are a small, documented set of canonical mesh indices. Check them by
plotting them on a face image (notebook 03 does this) before trusting any region result.
"""

from __future__ import annotations

import itertools
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from fel.imageio import load_rgb

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"
N_LANDMARKS = 478
N_BLENDSHAPES = 52

REGION_POINTS: dict[str, list[int]] = {
    "forehead": [10, 151, 9],
    "brow": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    "eye": [33, 133, 159, 145, 362, 263, 386, 374],
    "nose": [1, 4, 6, 168],
    "cheek": [50, 280, 234, 454],
    "lip": [61, 291, 0, 17, 13, 14, 78, 308],
    "chin": [152, 175, 199],
}
OUTER_EYE_LEFT, OUTER_EYE_RIGHT = 33, 263  # used to normalise for face size


def all_points() -> list[int]:
    return sorted({p for pts in REGION_POINTS.values() for p in pts})


def region_of_point(i: int) -> str:
    for name, pts in REGION_POINTS.items():
        if i in pts:
            return name
    raise KeyError(i)


def build_pairs() -> list[tuple[int, int]]:
    """Every unordered pair of selected landmarks, in a fixed order."""
    return list(itertools.combinations(all_points(), 2))


def pair_region(i: int, j: int) -> str:
    ri, rj = region_of_point(i), region_of_point(j)
    return ri if ri == rj else "cross"


def feature_names(pairs: Sequence[tuple[int, int]] | None = None) -> list[str]:
    return [f"d_{i}_{j}" for i, j in (pairs or build_pairs())]


def to_pixel_scale(landmarks_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """MediaPipe returns x, y in 0..1 and z on roughly the x scale. Convert to pixel scale."""
    out = np.asarray(landmarks_norm, dtype=float).copy()
    out[..., 0] *= width
    out[..., 1] *= height
    out[..., 2] *= width
    return out


def interocular(landmarks: np.ndarray) -> np.ndarray:
    d = landmarks[:, OUTER_EYE_LEFT, :] - landmarks[:, OUTER_EYE_RIGHT, :]
    return np.linalg.norm(d, axis=1)


def distance_features(landmarks: np.ndarray, use_z: bool = True) -> np.ndarray:
    """Pairwise distances between selected points, divided by the inter-ocular distance.

    landmarks: (n, 478, 3) in pixel scale. Dividing by the inter-ocular distance removes
    the effect of face size and image size. Head rotation is not corrected.
    """
    lm = np.asarray(landmarks, dtype=float)
    if lm.ndim != 3 or lm.shape[1] < max(all_points()) + 1:
        raise ValueError("landmarks must have shape (n, >=478, 3)")
    if not use_z:
        lm = lm[..., :2]
    pairs = build_pairs()
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    d = np.linalg.norm(lm[:, a, :] - lm[:, b, :], axis=2)
    scale = interocular(lm)
    if np.any(scale <= 0):
        raise ValueError("inter-ocular distance is zero for at least one sample")
    return d / scale[:, None]


def point_features(landmarks: np.ndarray, use_z: bool = True) -> np.ndarray:
    """Selected point coordinates, centred on the nose tip and scaled by inter-ocular distance."""
    lm = np.asarray(landmarks, dtype=float)
    if not use_z:
        lm = lm[..., :2]
    pts = all_points()
    centred = lm[:, pts, :] - lm[:, [1], :]
    return (centred / interocular(np.asarray(landmarks, dtype=float))[:, None, None]).reshape(
        len(lm), -1
    )


def ensure_model(dest_dir: str | Path = "data/cache") -> Path:
    """Download the Face Landmarker model file if it is not there. Prints a notice first."""
    dest = Path(dest_dir) / MODEL_FILENAME
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(
        "Downloading the MediaPipe Face Landmarker model from Google's model storage.\n"
        f"Source: {MODEL_URL}\n"
        "Read the model card for its terms before sharing anything derived from it."
    )
    urllib.request.urlretrieve(MODEL_URL, dest)
    return dest


class FaceExtractor:
    """Detect one face and return pixel-scale landmarks, blendshapes and a face box."""

    def __init__(self, model_path: str | Path):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "mediapipe is not installed. Run: pip install -e '.[landmarks]'"
            ) from exc
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract(self, image_path: str | Path):
        """Return (landmarks (478, 3), blendshapes (52,), box (x0, y0, x1, y1)) or None."""
        arr = load_rgb(image_path)
        height, width = arr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr)
        res = self._landmarker.detect(img)
        if not res.face_landmarks:
            return None
        lm = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=float)
        lm = to_pixel_scale(lm, width, height)
        bs = np.zeros(N_BLENDSHAPES)
        if res.face_blendshapes:
            for c in res.face_blendshapes[0]:
                bs[c.index] = c.score
        box = face_box(lm, width, height)
        return lm, bs, box


def face_box(
    landmarks: np.ndarray, width: int, height: int, margin: float = 0.1
) -> tuple[int, int, int, int]:
    """Bounding box of the landmarks with a relative margin, clipped to the image."""
    x0, y0 = landmarks[:, 0].min(), landmarks[:, 1].min()
    x1, y1 = landmarks[:, 0].max(), landmarks[:, 1].max()
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (
        int(max(0, x0 - mx)),
        int(max(0, y0 - my)),
        int(min(width, x1 + mx)),
        int(min(height, y1 + my)),
    )


def extract_many(paths: Sequence[str | Path], extract: Callable[[str | Path], object]):
    """Run `extract` over paths. Returns (ok_mask, landmarks, blendshapes, boxes).

    Rows where no face was found are left out of the arrays; ok_mask marks the kept rows.
    """
    ok, lms, bss, boxes = [], [], [], []
    for p in paths:
        r = extract(p)
        ok.append(r is not None)
        if r is not None:
            lms.append(r[0])
            bss.append(r[1])
            boxes.append(r[2])
    empty_lm = np.empty((0, N_LANDMARKS, 3))
    return (
        np.array(ok, dtype=bool),
        np.stack(lms) if lms else empty_lm,
        np.stack(bss) if bss else np.empty((0, N_BLENDSHAPES)),
        np.array(boxes, dtype=int) if boxes else np.empty((0, 4), dtype=int),
    )
