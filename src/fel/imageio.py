"""Face crops for the image model. Needs Pillow."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    """Open any image Pillow can read (PNG, JPEG, TIFF, ...) as an (H, W, 3) uint8 array."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required. Run: pip install -e '.[images]'") from exc
    with Image.open(path) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))


def load_gray_crop(path: str | Path, box: tuple[int, int, int, int], size: int = 48) -> np.ndarray:
    """Open an image, crop to the face box, convert to grey and resize to size x size (uint8)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required. Run: pip install -e '.[images]'") from exc
    with Image.open(path) as im:
        crop = im.convert("L").crop(box).resize((size, size), Image.BILINEAR)
        return np.asarray(crop, dtype=np.uint8)
