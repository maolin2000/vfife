"""The four places this project states its version, and the one that lies.

A release states its number in four files, and nothing checks that they agree:
`pyproject.toml` (what pip installs), `vfife/__version__` (what the code
reports), `CITATION.cff` (what a citation says), and the top released heading
of `CHANGELOG.md` (what the history says). Three of them can be bumped and the
fourth forgotten, and every test still passes -- the package then ships one
number and reports another, which is the kind of untruth v0.3's
release audit was written to catch and which it caught by hand.

This is the cheapest invariant in the repository: four strings, one assertion.
"""
import pathlib
import re

import pytest

import vfife

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: x.y.z, or a PEP 440 pre-release of it (1.0.0rc1) -- the 1.0 release
#: candidates are numbered so pip orders them before 1.0.0
SEMVER = re.compile(r"^\d+\.\d+\.\d+((a|b|rc)\d+)?$")


def _pyproject():
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        m = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    raise AssertionError("pyproject.toml has no `version = \"...\"` line")


def _citation():
    for line in (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*version\s*:\s*(\S+)", line)
        if m:
            return m.group(1).strip('"\'')
    raise AssertionError("CITATION.cff has no `version:` line")


def _changelog():
    """The newest RELEASED heading: `## [0.3.0] - date`, skipping Unreleased."""
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"##\s*\[([^\]]+)\]", line)
        if m and m.group(1).lower() != "unreleased":
            return m.group(1)
    raise AssertionError("CHANGELOG.md has no released `## [x.y.z]` heading")


def test_the_four_version_strings_agree():
    stated = {"vfife.__version__": vfife.__version__,
              "pyproject.toml": _pyproject(),
              "CITATION.cff": _citation(),
              "CHANGELOG.md": _changelog()}
    assert len(set(stated.values())) == 1, (
        "the version is stated in four places and they disagree:\n  "
        + "\n  ".join(f"{k:<20} {v}" for k, v in stated.items())
        + "\nBump all four, or the package ships one number and reports another.")


@pytest.mark.parametrize("name,getter", [
    ("vfife.__version__", lambda: vfife.__version__),
    ("pyproject.toml", _pyproject),
    ("CITATION.cff", _citation),
    ("CHANGELOG.md", _changelog)])
def test_each_one_is_a_semantic_version(name, getter):
    value = getter()
    assert SEMVER.match(value), f"{name} states {value!r}, which is not x.y.z (or x.y.zrcN)"


def test_the_changelog_has_an_unreleased_section_above_it():
    """Work lands under `## [Unreleased]` and is renamed at release time.
    Without the section, a stage's entry has nowhere to go and
    quietly ends up under the last release, which then describes work it did
    not contain."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^##\s*\[Unreleased\]", text, re.M), (
        "CHANGELOG.md has no `## [Unreleased]` section; the next stage's entry "
        "has nowhere to land")
