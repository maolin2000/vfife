"""1.1, item 3: `vfife summary`.

The page is read for one check above all: the model weighs what it should,
and the reactions carry it. A column under its own weight and a line load
with its mass: mass = rho A H + q H / g, weight = rho A H g, line load = q H,
and the vertical reaction over (weight + line load) is 1.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_added_mass import STEEL, _column_case
from test_member_failure import _run
from vfife.post.summary import summarize, text

G = 9.80665
Q = 2.0e4
ROOT = Path(__file__).resolve().parents[1]


def _column(tmp_path):
    mons = {"rz": {"kind": "reaction", "region": "base", "dof": "z"}}
    loads = [{"region": "all", "gravity": [0, 0, -G]},
             {"region": "col", "line_load": [0.0, 0.0, -Q], "with_mass": True},
             {"region": "top", "vector": [500.0, 0, 0]}]
    _p, d = _column_case(tmp_path, "sc", loads=loads, material=STEEL, monitors=mons)
    d["output"].pop("fields")                 # the default fields (1.1): reactions, forces
    _out, h5 = _run(tmp_path, "sc", d)
    return h5


def test_the_model_weighs_what_it_should_and_the_reactions_carry_it(tmp_path):
    h5 = _column(tmp_path)
    s = summarize(h5)
    own = 7850.0 * 1e-2 * 4.0
    assert s["totals_from"] == "result"
    np.testing.assert_allclose(s["totals"]["mass"], own + Q * 4.0 / G, rtol=1e-12)
    np.testing.assert_allclose(s["totals"]["weight"], [0, 0, -own * G], rtol=1e-12)
    np.testing.assert_allclose(s["totals"]["line_load"], [0, 0, -Q * 4.0], rtol=1e-12)
    assert s["reaction_last"][2] == pytest.approx(own * G + Q * 4.0, rel=1e-6)
    page = text(s)
    assert "reaction z / (weight + line and point loads) = 1.0000" in page
    assert s["status"] == "CONVERGED" and s["converged"] is True


def test_the_largest_values_say_where_and_when(tmp_path):
    s = summarize(_column(tmp_path))
    ext = s["extremes"]
    # the top pushed sideways: the largest |u| at the top, x
    assert ext["displacement"]["node"] == 4
    np.testing.assert_allclose(ext["displacement"]["at"], [0, 0, 4.0])
    assert ext["displacement_components"][0]["axis"] == "x"
    # the column is in compression from the weight above it: the axial
    # force's minimum is at the base member, about the whole load
    lo = ext["axial_force"]["min"]
    assert lo["element"] == 0 and lo["value"] < 0
    # the sideways push bends it about one axis; a force zero throughout is left out
    assert ("bending_y" in ext) != ("bending_z" in ext)
    assert "torsion" not in ext
    assert s["monitors"]["rz"]["final"] == pytest.approx(s["reaction_last"][2], rel=1e-6)


def test_a_result_from_before_1_1_takes_the_mass_from_its_case():
    h5 = ROOT / "tests" / "data" / "result_format_2" / "cantilever.h5"
    s = summarize(h5)
    assert not s["totals"]
    assert s["time_axis"] == "time"            # format 2 had no time_axis: the mode says it
    s = summarize(h5, case=str(ROOT / "tests" / "data" / "result_format_2" / "cantilever.yaml"))
    assert s["totals_from"] == "case" and s["totals"]["mass"][0] > 0


def test_the_command_prints_the_page_and_json(tmp_path):
    h5 = _column(tmp_path)
    page = subprocess.run([sys.executable, "-m", "vfife", "summary", str(h5)], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8")
    assert page.returncode == 0, page.stderr
    assert "what the model weighs and carries" in page.stdout and "largest over all frames" in page.stdout
    js = subprocess.run([sys.executable, "-m", "vfife", "summary", str(h5), "--json"], cwd=ROOT,
                        capture_output=True, text=True, encoding="utf-8")
    assert js.returncode == 0, js.stderr
    assert json.loads(js.stdout)["frames"] >= 1


def test_a_step_s_own_loads_count_and_the_page_refuses_what_is_not_a_run(tmp_path):
    """The loads beside the last frame's reactions are the last step's: a
    gravity load written inside `steps` is weighed (review of 1.1)."""
    from vfife.core.diagnostics import UserError
    mons = {"rz": {"kind": "reaction", "region": "base", "dof": "z"}}
    _p, d = _column_case(tmp_path, "st", material=STEEL, monitors=mons)
    d["output"].pop("fields")
    d["steps"] = [{"name": "dead", "loads": [{"region": "all", "gravity": [0, 0, -G]}]}]
    _out, h5 = _run(tmp_path, "st", d)
    s = summarize(h5)
    own = 7850.0 * 1e-2 * 4.0
    np.testing.assert_allclose(s["totals"]["weight"], [0, 0, -own * G], rtol=1e-12)
    assert "= 1.0000" in text(s)
    with pytest.raises(UserError, match="checkpoint"):
        summarize(str(h5).replace(".h5", ".ckpt.h5"))


def test_json_has_no_nan():
    from vfife.post.summary import _finite
    out = json.dumps(_finite({"a": float("nan"), "b": [1.0, float("inf")], "c": np.float32(2.0)}))
    assert json.loads(out) == {"a": None, "b": [1.0, None], "c": 2.0}
