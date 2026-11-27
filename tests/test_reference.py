"""docs/yaml_reference.md is generated from the code and must match it."""
import pathlib
import os
import subprocess
import sys

import pytest

from vfife import reference
from vfife.io import model as im
from vfife.io import schema as sc

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_reference_on_disk_is_what_the_code_generates():
    on_disk = (ROOT / "docs" / "yaml_reference.md").read_text(encoding="utf-8")
    assert on_disk == reference.generate(), \
        "docs/yaml_reference.md is stale: run `python -m vfife reference`"


def test_every_field_of_the_input_model_has_a_doc_comment():
    seen = set()
    missing = []

    def visit(cls, path):
        if cls in seen:
            return
        seen.add(cls)
        docs = reference.field_docs(cls)
        for name, f in cls.model_fields.items():
            if not docs.get(name):
                missing.append(f"{path}.{name}")
            fpath = f"{path}.{name}"
            inner, _ = reference._unwrap(f.annotation)
            if inner is not None:
                visit(inner, fpath)
            elif fpath in reference.EXTRA:
                visit(reference.EXTRA[fpath], fpath)

    for block, cls, _ in reference.BLOCKS:
        if cls is not None:
            visit(cls, block)
    assert not missing, "fields without a full `#:` comment: " + ", ".join(missing)


def test_usage_finds_the_features_the_shipped_cases_are_known_to_use():
    hits, cases = reference.usage(ROOT / "cases")
    assert "26" in hits["loads[].moving_load.axles"]           # Fryba's crossing force
    assert "12b" in hits["material:bond_ceb"]                  # an inline bond law
    assert "24c" in hits["sections.*.tendons"]                 # fibre-section tendons
    assert "05" in hits["loads[].ground_motion"]
    assert "12b" in hits["assign[].bond.type"]               # a dict-typed sub-block
    assert len(cases) >= 50


def test_the_cli_check_reports_a_stale_file(tmp_path):
    target = tmp_path / "ref.md"
    target.write_text("stale", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "vfife", "reference", "--check", "-o", str(target)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 1 and "out of date" in r.stdout
    r = subprocess.run([sys.executable, "-m", "vfife", "reference", "-o", str(target)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0 and target.read_text(encoding="utf-8") == reference.generate()
