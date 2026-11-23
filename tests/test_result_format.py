"""The case and result formats (route 0.9 item 1).

A result file written by this version is kept in tests/data/result_format_2
and every reader must go on reading it: the monitors, `verify`, the 3-D view,
`--restart` and `initial_state`. A file that says it is newer is refused by
name; one that predates the format attribute is still read.

The case format: `version: 1`, a larger number refused; the old spellings of
section keys read with a warning, and `vfife migrate` rewriting them where
they stand -- comments and layout untouched -- after proving the model is
the same.
"""
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case, run_case
from vfife.core.diagnostics import UserError
from vfife.io.writer import RESULT_FORMAT, read_checkpoint
from vfife.post import read_monitors

DATA = Path(__file__).resolve().parent / "data" / "result_format_2"
W_TIP = [-4.1319e-4, -1.20402e-3, -1.95328e-3, -2.16961e-3]


def _copy(tmp_path):
    for f in DATA.iterdir():
        shutil.copy(f, tmp_path / f.name)
    return tmp_path / "cantilever.yaml", tmp_path / "cantilever.h5"


def test_the_kept_result_is_read_by_every_reader(tmp_path):
    case, h5 = _copy(tmp_path)
    with h5py.File(h5) as f:
        assert int(f.attrs["result_schema_version"]) == 2 <= RESULT_FORMAT
    # monitors
    m = read_monitors(str(h5))
    assert m["w_tip"].values == pytest.approx(W_TIP, rel=1e-5)
    # verify
    from vfife.verification import verify
    contract = tmp_path / "contract.yaml"
    contract.write_text(yaml.safe_dump({
        "version": 1, "case": "result_format_2", "classification": "regression",
        "reference": "the kept result itself",
        "metrics": [{"id": "tip", "description": "tip deflection at 0.02 s", "source": "monitor",
                     "name": "w_tip", "reduction": "final",
                     "expect": {"target": -2.16961e-3, "rtol": 1e-4}}]}), encoding="utf-8")
    assert verify(contract, h5)["status"] == "PASS"
    # the 3-D view
    from vfife.post import view
    act = view.load([str(h5)])
    assert act is not None
    # the checkpoint is read; an initial_state starts from it. A RESTART of
    # it is refused by name: it was written before checkpoints carried the
    # loop position, and resuming it would replay the run
    # from step 0 on top of the state it reached
    ck = read_checkpoint(str(h5))
    assert int(np.asarray(ck["step"]).ravel()[0]) == 674
    with pytest.raises(UserError) as e:
        run_case(str(case), output=str(tmp_path / "restarted.h5"), restart=str(h5))
    assert e.value.code == "RESTART_OLD_CHECKPOINT"
    assert "initial_state" in e.value.hint
    d = yaml.safe_load(case.read_text(encoding="utf-8"))
    d["solver"]["initial_state"] = str(h5)
    d["solver"]["total_time"] = 0.002
    (tmp_path / "from_state.yaml").write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    out = run_case(str(tmp_path / "from_state.yaml"), output=str(tmp_path / "from_state.h5"))
    assert out.status == "COMPLETED_DYNAMIC"
    with h5py.File(tmp_path / "from_state.h5") as f:
        # it started at the kept state: a tip already near -2.17 mm, not 0
        assert f["monitors/w_tip"][0] < -1.5e-3


def test_a_newer_result_is_refused_by_name_and_an_older_one_read(tmp_path):
    _case, h5 = _copy(tmp_path)
    with h5py.File(h5, "a") as f:
        f.attrs["result_schema_version"] = RESULT_FORMAT + 1
    for read in (read_monitors, read_checkpoint):
        with pytest.raises(UserError) as e:
            read(str(h5))
        assert e.value.code == "RESULT_TOO_NEW"
    with h5py.File(h5, "a") as f:
        del f.attrs["result_schema_version"]           # before the attribute existed
    assert read_monitors(str(h5))["w_tip"].values == pytest.approx(W_TIP, rel=1e-5)


def test_a_newer_case_format_is_refused(tmp_path):
    case, _h5 = _copy(tmp_path)
    case.write_text(case.read_text(encoding="utf-8").replace("version: 1", "version: 2"), encoding="utf-8")
    with pytest.raises(Exception, match="newer case format"):
        prepare_case(str(case))


def test_old_spellings_are_read_warned_and_migrated_in_place(tmp_path):
    from vfife.cli import main
    case, _h5 = _copy(tmp_path)
    text = case.read_text(encoding="utf-8")
    old = (text.replace("version: 1\n", "")
               .replace("width: 0.1, height: 0.1}", "b: 0.1, h: 0.1}  # the old spelling"))
    case.write_text(old, encoding="utf-8")
    p = prepare_case(str(case))
    codes = [i["code"] for i in p.preflight["issues"]]
    assert codes.count("DEPRECATED_FIELD") == 2
    assert p.preflight["status"] == "PASS"
    # migrate: renamed where they stand, the comment kept, version added
    main(["migrate", str(case), "--in-place"])
    new = case.read_text(encoding="utf-8")
    assert "width: 0.1, height: 0.1}  # the old spelling" in new
    assert new.startswith(text.split("version: 1")[0] + "version: 1\n")
    assert [i["code"] for i in prepare_case(str(case)).preflight["issues"]].count("DEPRECATED_FIELD") == 0
    # and the model is the one it was
    from vfife.io.migrate import same_model
    assert same_model(old, new) == []


def test_old_and_new_spelling_together_must_agree():
    from vfife.io.model import SectionSpec
    assert SectionSpec(type="beam", material="s", shape="rect", b=0.1, width=0.1, height=0.2).width == 0.1
    with pytest.raises(Exception, match="both given"):
        SectionSpec(type="beam", material="s", shape="rect", b=0.1, width=0.3, height=0.2)


def test_j_now_reaches_a_fibre_section():
    """`torsion_constant` was a fibre section's name for a beam's `j`, and a
    `j` written on a fibre section was dropped without a word. Now both
    reach it."""
    from vfife.io.model import InputCase
    base = {"materials": {"c": {"type": "linear_elastic", "E": 3e10, "nu": 0.2, "rho": 2500}},
            "sections": {"f": {"type": "fiber", "material": "c", "shape": "rect",
                               "width": 0.3, "height": 0.5}},
            "assign": [{"region": "all", "section": "f"}],
            "name": "x", "mesh": {"type": "gmsh", "path": "x.msh"}}
    for key in ("j", "torsion_constant"):
        d = yaml.safe_load(yaml.safe_dump(base))
        d["sections"]["f"][key] = 0.0123
        low = InputCase.model_validate(d).lower()
        assert low["element"]["fiber_section"]["torsion_constant"] == 0.0123
