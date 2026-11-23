"""1.2, item 7: `vfife new`, scenario templates that run as they stand.

Every template is written, checked, run, and its summary's weight check
read: the reactions carry what the model weighs and the loads it is given,
to 1e-4, which a template with a wrong section, support or load would not.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from vfife.api import run_case
from vfife.post.summary import summarize, text
from vfife.scaffold import TEMPLATES

ROOT = Path(__file__).resolve().parents[1]


def _cli(*args, cwd):
    return subprocess.run([sys.executable, "-m", "vfife", *args], cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8")


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_checks_runs_and_carries_its_loads(tmp_path, name):
    out = _cli("new", name, f"{name}.yaml", cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    case = tmp_path / f"{name}.yaml"
    first = case.read_text(encoding="utf-8").splitlines()[0]
    assert first == "# yaml-language-server: $schema=vfife.schema.json"
    assert (tmp_path / "vfife.schema.json").exists()
    chk = _cli("check", case.name, cwd=tmp_path)
    assert chk.returncode == 0, chk.stdout + chk.stderr
    import os
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        res = run_case(str(case))
    finally:
        os.chdir(old)
    assert res.converged, res.message
    page = text(summarize(tmp_path / "results" / f"{name}.h5"))
    ratio = float(page.split(" = ")[1].split()[0]) if " = " in page else None
    assert ratio == pytest.approx(1.0, abs=1e-4), page


def test_new_lists_refuses_and_keeps(tmp_path):
    out = _cli("new", "--list", cwd=tmp_path)
    assert out.returncode == 0 and all(k in out.stdout for k in TEMPLATES)
    bad = _cli("new", "girdr", cwd=tmp_path)
    assert bad.returncode == 1 and "NEW_UNKNOWN_TEMPLATE" in bad.stdout + bad.stderr
    assert "did you mean 'girder'" in bad.stdout + bad.stderr
    assert _cli("new", "girder", cwd=tmp_path).returncode == 0
    again = _cli("new", "girder", cwd=tmp_path)
    assert again.returncode == 1 and "NEW_EXISTS" in again.stdout + again.stderr
    assert _cli("new", "girder", "--overwrite", cwd=tmp_path).returncode == 0


@pytest.mark.parametrize("stem", ["2024", "yes", "#1"])
def test_a_case_named_like_a_number_or_a_word_yaml_reads_is_still_a_case(tmp_path, stem):
    out = _cli("new", "girder", f"{stem}.yaml", cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    info = _cli("info", f"{stem}.yaml", cwd=tmp_path)
    assert info.returncode == 0, info.stdout + info.stderr


def test_a_case_named_without_an_extension_gets_yaml(tmp_path):
    """`vfife new girder g1` wrote a file named `g1` that `vfife run g1`
    then had to be told was a case (1.4.1); a name with an extension, or a
    folder, is left as it is."""
    out = _cli("new", "girder", "g1", cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert (tmp_path / "g1.yaml").exists() and not (tmp_path / "g1").exists()
    assert _cli("new", "girder", "g2.yml", cwd=tmp_path).returncode == 0
    assert (tmp_path / "g2.yml").exists()
    (tmp_path / "sub").mkdir()
    folder = _cli("new", "girder", "sub", cwd=tmp_path)
    assert folder.returncode == 1 and "NEW_EXISTS" in folder.stdout + folder.stderr
