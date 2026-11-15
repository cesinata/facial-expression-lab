import re
from pathlib import Path

from fel.fer import FERPLUS_CLASSES

ROOT = Path(__file__).resolve().parents[1]


def test_demo_class_list_matches_python_class_order():
    js = (ROOT / "demo" / "app.js").read_text()
    m = re.search(r"const CLASSES = \[(.*?)\];", js, re.S)
    assert m, "CLASSES not found in demo/app.js"
    assert re.findall(r'"([a-z]+)"', m.group(1)) == FERPLUS_CLASSES


def test_emoji_mirror_has_one_emoji_per_class():
    js = (ROOT / "demo" / "app.js").read_text()
    m = re.search(r"const EMOJI = \{(.*?)\};", js, re.S)
    assert m, "EMOJI not found in demo/app.js"
    pairs = re.findall(r'([a-z]+): "(\\u\{[0-9A-F]+\})"', m.group(1))
    assert [k for k, _ in pairs] == FERPLUS_CLASSES
    assert len({v for _, v in pairs}) == len(FERPLUS_CLASSES)


def test_emoji_mirror_has_a_smoothing_control_and_disclaimer():
    html = (ROOT / "demo" / "index.html").read_text()
    assert 'id="smooth"' in html and 'id="mirror"' in html
    assert "Not what you feel" in html


def test_demo_face_box_margin_matches_python():
    js = (ROOT / "demo" / "app.js").read_text()
    assert "const MARGIN = 0.1;" in js


def test_demo_cdn_urls_are_pinned_to_exact_versions():
    text = (ROOT / "demo" / "app.js").read_text() + (ROOT / "demo" / "index.html").read_text()
    urls = re.findall(r"https://cdn\.jsdelivr\.net/npm/[^\"'`\s]+", text)
    assert urls
    for u in urls:
        assert re.search(r"@\d+\.\d+\.\d+/", u), f"not pinned: {u}"
        assert "@latest" not in u


def test_demo_model_input_name_matches_the_export():
    js = (ROOT / "demo" / "app.js").read_text()
    src = (ROOT / "src" / "fel" / "train_fer.py").read_text()
    assert 'input_names=["input"]' in src and "session.run({ input:" in js
    assert 'output_names=["logits"]' in src and "out.logits" in js


def test_demo_uses_one_version_of_each_cdn_library():
    text = (ROOT / "demo" / "app.js").read_text() + (ROOT / "demo" / "index.html").read_text()
    for lib in ("onnxruntime-web", "@mediapipe/tasks-vision"):
        versions = set(re.findall(re.escape(lib) + r"@(\d+\.\d+\.\d+)/", text))
        assert len(versions) == 1, f"{lib} pinned to several versions: {versions}"


def test_demo_page_says_where_the_model_comes_from():
    html = (ROOT / "demo" / "index.html").read_text()
    assert "GitHub release" in html and "FP32" in html
