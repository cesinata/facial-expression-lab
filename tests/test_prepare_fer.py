import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_fer_data",
                                              ROOT / "scripts" / "prepare_fer_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

CSV = b"emotion,pixels,Usage\n0,1 2 3,Training\n"


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


def _tar_bytes(name, data):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _ferplus(tmp_path):
    p = tmp_path / "FERPlus-master.zip"
    _zip(p, {"FERPlus-master/fer2013new.csv": b"Usage,neutral\nTraining,1\n",
             "FERPlus-master/LICENSE.md": b"Some licence text\n"})
    return p


def test_kaggle_zip_with_plain_csv(tmp_path):
    k = tmp_path / "fer2013.zip"
    _zip(k, {"fer2013.csv": CSV})
    paths = prep.prepare(k, _ferplus(tmp_path), tmp_path / "out")
    assert paths["fer_csv"].endswith("fer2013.csv") and paths["ferplus_csv"].endswith("fer2013new.csv")
    assert paths["license_path"].endswith("LICENSE.md")
    assert json.loads((tmp_path / "out" / "paths.json").read_text()) == paths


def test_kaggle_file_can_be_a_tar_gz_directly(tmp_path):
    k = tmp_path / "fer2013.tar.gz"
    k.write_bytes(_tar_bytes("fer2013/fer2013.csv", CSV))
    paths = prep.prepare(k, _ferplus(tmp_path), tmp_path / "out")
    assert paths["fer_csv"].endswith("fer2013.csv")


def test_kaggle_zip_with_a_tar_gz_inside(tmp_path):
    k = tmp_path / "fer2013.zip"
    _zip(k, {"fer2013.tar.gz": _tar_bytes("fer2013/fer2013.csv", CSV), "other.txt": b"x"})
    paths = prep.prepare(k, _ferplus(tmp_path), tmp_path / "out")
    assert paths["fer_csv"].endswith("fer2013.csv")


def test_kaggle_zip_with_icml_csv_inside_a_nested_zip(tmp_path):
    inner = tmp_path / "inner.zip"
    _zip(inner, {"icml_face_data.csv": b"emotion, Usage, pixels\n0,Training,1 2\n"})
    k = tmp_path / "fer2013.zip"
    _zip(k, {"icml_face_data.csv.zip": inner.read_bytes()})
    paths = prep.prepare(k, _ferplus(tmp_path), tmp_path / "out")
    assert paths["fer_csv"].endswith("icml_face_data.csv")


def test_fer2013_csv_is_preferred_over_icml(tmp_path):
    k = tmp_path / "fer2013.zip"
    _zip(k, {"icml_face_data.csv": CSV, "sub/fer2013.csv": CSV})
    paths = prep.prepare(k, _ferplus(tmp_path), tmp_path / "out")
    assert paths["fer_csv"].endswith("fer2013.csv")


def test_image_folder_copy_gets_a_specific_hint(tmp_path):
    k = tmp_path / "fer2013.zip"
    _zip(k, {f"train/angry/{i}.jpg": b"x" for i in range(120)})
    with pytest.raises(FileNotFoundError) as e:
        prep.prepare(k, _ferplus(tmp_path), tmp_path / "out")
    assert "image-folder copy" in str(e.value)


def test_missing_files_give_a_helpful_error(tmp_path):
    k = tmp_path / "fer2013.zip"
    _zip(k, {"readme.txt": b"nothing useful"})
    with pytest.raises(FileNotFoundError) as e:
        prep.prepare(k, _ferplus(tmp_path), tmp_path / "out")
    assert "readme.txt" in str(e.value)
