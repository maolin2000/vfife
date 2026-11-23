"""The English manual follows the Chinese one section for section.
The Chinese manual is the reference; this catches a
section added to one and not the other, which is how translations drift."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _sections(name):
    out = []
    for line in (ROOT / "docs" / name).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(#+)\s+(附录\s*[A-Z]|Appendix\s+[A-Z]|[0-9]+(?:\.[0-9]+)*)", line)
        if m:
            out.append((len(m.group(1)), re.sub(r"附录\s*|Appendix\s+", "App ", m.group(2))))
    return out


def _fences(name):
    return sum(1 for l in (ROOT / "docs" / name).read_text(encoding="utf-8").splitlines()
               if l.lstrip().startswith("```"))


def test_the_english_manual_has_the_chinese_manuals_sections():
    zh, en = _sections("用户手册.md"), _sections("user_manual.md")
    assert len(zh) > 60
    missing = [s for s in zh if s not in en]
    extra = [s for s in en if s not in zh]
    assert zh == en, f"sections only in the Chinese manual: {missing}; only in the English: {extra}"


def test_the_english_manual_has_the_same_examples():
    assert _fences("用户手册.md") == _fences("user_manual.md")
