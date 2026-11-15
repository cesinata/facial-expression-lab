"""Every relative Markdown link in README.md and docs/*.md must point to a file that exists."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
# [text](target) or [text](target "title"); images ![alt](target) match too.
LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
EXTERNAL = ("http://", "https://", "mailto:", "#")
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)  # fenced blocks and inline code


def relative_links(text: str) -> list[str]:
    """Targets of relative links outside code, with any #anchor or ?query removed."""
    out = []
    for target in LINK.findall(CODE.sub("", text)):
        if target.startswith(EXTERNAL):
            continue
        path = re.split(r"[#?]", target, maxsplit=1)[0]
        if path:
            out.append(path)
    return out


def test_relative_links_parser_skips_external_links_and_anchors():
    text = ("[a](docs/x.md) [b](https://example.org/y) [c](#top) [d](../src/z.py#L3) "
            '[e](mailto:someone@example.org) [f](docs/w.md "title")')
    assert relative_links(text) == ["docs/x.md", "../src/z.py", "docs/w.md"]


def test_relative_links_parser_ignores_links_inside_code():
    text = "Syntax: `[text](target)`.\n```\n[y](missing.md)\n```\n[real](docs/r.md)"
    assert relative_links(text) == ["docs/r.md"]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_point_to_existing_files(doc):
    text = doc.read_text(encoding="utf-8")
    missing = [t for t in relative_links(text) if not (doc.parent / t).exists()]
    assert not missing, f"{doc.relative_to(ROOT)} links to missing files: {missing}"
