"""The README is the front page (since 1.12; the long description moved to
docs/technical_notes.md). What a first reader copies from it has to work:

* its case is the one `vfife new girder` writes, and it runs to the numbers
  a hand check gives;
* every command it shows exists;
* every link in it, and in the technical notes, leads to a file;
* it says the version the package has, and stays a front page.
"""
import pathlib
import re

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _blocks(lang):
    return re.findall(rf"```{lang}\n(.*?)```", README, flags=re.S)


def test_the_case_on_the_front_page_is_the_girder_template_and_runs(tmp_path):
    from vfife.api import run_case
    (block,) = _blocks("yaml")
    d = yaml.safe_load(block)
    text = (ROOT / "vfife" / "templates" / "girder.yaml").read_text(encoding="utf-8")
    template = yaml.safe_load(text.replace("@NAME@", "girder"))
    d["output"]["path"] = template["output"]["path"] = str(tmp_path / "girder.h5")
    assert d == template, "the README's case and `vfife new girder` have drifted apart"
    out = run_case(d)
    assert out.status == "CONVERGED", out.message
    # the four supports carry the weight, 100 m of (rho A g + 60 kN/m); B takes about a third
    W = 100.0 * (7850.0 * 0.35 * 9.80665 + 6.0e4)
    assert 0.3 * W < float(np.asarray(out.monitors["R_B"])[-1]) < 0.45 * W
    assert -0.01 < float(np.asarray(out.monitors["w_main"])[-1]) < 0.0


def test_every_command_on_the_front_page_exists(capsys):
    from vfife import cli
    used = set(re.findall(r"^vfife (\w+)", "\n".join(_blocks("bash")), flags=re.M))
    used |= set(re.findall(r"`vfife (\w+)", README))
    assert {"new", "info", "run", "summary", "modal", "view", "probe", "mesh", "init", "schema"} <= used
    for cmd in sorted(used):
        with pytest.raises(SystemExit) as stop:
            cli.main([cmd, "--help"])
        assert stop.value.code == 0, cmd
    capsys.readouterr()


@pytest.mark.parametrize("doc", ["README.md", "docs/technical_notes.md", "cases/README.md"])
def test_every_relative_link_leads_to_a_file(doc):
    path = ROOT / doc
    text = path.read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if not (path.parent / target.split("#")[0]).exists():
            missing.append(target)
    assert not missing, missing


def test_the_front_page_names_the_version_and_stays_short():
    import vfife
    assert f"**Version {vfife.__version__}**" in README
    assert f"vfife-{vfife.__version__}-py3-none-any.whl" in README
    assert len(README.splitlines()) < 250, "the long description belongs in docs/technical_notes.md"
    n = len(list((ROOT / "cases").glob("*.yaml")))
    assert f"The {n} reference cases" in README
