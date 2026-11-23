"""The package runs on Python 3.10 and 3.11 (pyproject: requires-python >=3.10).

Python 3.12 (PEP 701) lets an f-string's expression reuse the f-string's own
quote -- f"{d.get("a")}" -- and 3.10/3.11 refuse the whole module with a
SyntaxError. Every machine the library is developed and checked on runs 3.12
or later, so nothing local catches it: the 1.2.0 fibre solver had two such
lines, and CI on 3.11 failed 486 tests at import. This
test reads every module's tokens and refuses a string inside an f-string's
braces that uses the enclosing f-string's quote.
"""
import io
import pathlib
import sys
import tokenize

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIRS = ("vfife", "tests", "tools", "benchmarks")


def _files():
    out = [ROOT / "conftest.py"]
    for d in DIRS:
        out += sorted((ROOT / d).rglob("*.py"))
    return [p for p in out if p.exists()]


def _quote(tok_string):
    s = tok_string.lstrip("rRbBuUfF")
    return s[:3] if s[:3] in ('"""', "'''") else s[:1]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="older Pythons refuse such a module at import already")
def test_no_f_string_reuses_its_own_quote():
    bad = []
    for path in _files():
        stack = []
        # line endings as Python reads them (a CRLF file, a stray CR)
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            name = tokenize.tok_name[tok.type]
            if name == "FSTRING_START":
                q = _quote(tok.string)
                if stack and q == stack[-1]:
                    bad.append(f"{path.relative_to(ROOT)}:{tok.start[0]}")
                stack.append(q)
            elif name == "FSTRING_END":
                stack.pop()
            elif name == "STRING" and stack and _quote(tok.string) == stack[-1]:
                bad.append(f"{path.relative_to(ROOT)}:{tok.start[0]}")
    assert not bad, "f-strings that reuse their own quote (a SyntaxError before Python 3.12): " + ", ".join(bad)
