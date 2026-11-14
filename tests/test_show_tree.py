import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("show_tree", ROOT / "scripts" / "show_tree.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def _make(tmp_path):
    (tmp_path / "data" / "train" / "angry").mkdir(parents=True)
    (tmp_path / "data" / "train" / "angry" / "a.jpg").write_bytes(b"x")
    (tmp_path / "data" / "train" / "angry" / "b.jpg").write_bytes(b"x")
    (tmp_path / "data" / "notes.txt").write_text("hello")
    with zipfile.ZipFile(tmp_path / "data" / "fer2013.zip", "w") as z:
        z.writestr("train/angry/1.jpg", b"x")
        z.writestr("train/happy/2.jpg", b"x")
        z.writestr("fer2013.tar.gz", b"not a real archive")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        info = tarfile.TarInfo("fer2013/fer2013.csv")
        info.size = 3
        t.addfile(info, io.BytesIO(b"abc"))
    (tmp_path / "data" / "fer2013.tar.gz").write_bytes(buf.getvalue())
    return tmp_path / "data"


def test_zip_summary_counts_extensions_and_flags_nested_archives(tmp_path):
    root = _make(tmp_path)
    text = "\n".join(st.describe(root / "fer2013.zip"))
    assert "files: 3" in text and "'.jpg': 2" in text
    assert "nested archives: ['fer2013.tar.gz']" in text
    assert "top-level entries (2)" in text


def test_tar_summary_lists_the_csv(tmp_path):
    root = _make(tmp_path)
    text = "\n".join(st.describe(root / "fer2013.tar.gz"))
    assert "fer2013/fer2013.csv" in text and "'.csv': 1" in text


def test_directory_view_shows_archives_inside_and_respects_depth(tmp_path):
    root = _make(tmp_path)
    text = "\n".join(st.describe(root, depth=1))
    assert "[archive] fer2013.zip" in text and "notes.txt" in text
    assert "folders below depth 1 not shown" in text
    deeper = "\n".join(st.describe(root, depth=3))
    assert "angry/" in deeper


def test_missing_path_and_bad_archive_do_not_crash(tmp_path):
    assert st.describe(tmp_path / "nope")[0].startswith("not found")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    assert "could not open" in "\n".join(st.describe(bad))
