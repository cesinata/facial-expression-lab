import numpy as np
import pytest

from fel.imageio import load_gray_crop, load_rgb


@pytest.mark.parametrize("name,mode", [
    ("rgb.png", "RGB"), ("rgb.tiff", "RGB"), ("gray.png", "L"), ("gray.tiff", "L"),
])
def test_load_rgb_returns_h_w_3_uint8_for_png_and_tiff(tmp_path, name, mode):
    Image = pytest.importorskip("PIL.Image")
    shape = (30, 20, 3) if mode == "RGB" else (30, 20)
    arr = (np.random.default_rng(1).random(shape) * 255).astype(np.uint8)
    Image.fromarray(arr, mode).save(tmp_path / name)
    out = load_rgb(tmp_path / name)
    assert out.shape == (30, 20, 3) and out.dtype == np.uint8
    if mode == "L":
        assert (out[..., 0] == arr).all() and (out[..., 1] == arr).all()


def test_load_gray_crop_shape_and_type(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    arr = (np.random.default_rng(0).random((200, 100, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(tmp_path / "x.png")
    out = load_gray_crop(tmp_path / "x.png", (10, 20, 90, 180), size=48)
    assert out.shape == (48, 48) and out.dtype == np.uint8
