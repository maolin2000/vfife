"""Midas' creep and shrinkage of concrete translated (1.22,
io/midas_case/time_effects.py)."""
import numpy as np
import pytest
import yaml

from vfife.io import midas, midas_case

E = 3.45e7            # kN/m2
FCU = 50000.0         # kN/m2, C50's cube strength
MCT = """*VERSION
   9.1.0
*UNIT
   KN   , M, KJ, C
*NODE
     1, 0, 0, 0
     2, 5, 0, 0
     3, 10, 0, 0
*ELEMENT
     1, BEAM  ,    1,     1,     1,     2,     0,     0
     2, BEAM  ,    1,     1,     2,     3,     0,     0
*GROUP
   梁      , 1to3, 1to2, 0
*BNDR-GROUP
   支座
*LOAD-GROUP
   自重
*MATERIAL
    1, CONC, C50 , 0, 0, , C, NO, 0.05, 2,  3.4500e+07,   0.2,  1.0000e-05, 25,     0
*SECTION
    1, DBUSER    , 方1x1         , CC, 0, 0, 0, 0, 0, 0, YES, NO, SB  , 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
*STLDCASE
   自重 , CS,
*CONSTRAINT
   1, 111111, 支座
*USE-STLD, 自重
*SELFWEIGHT
0, 0, -1, 自重
*TDM-TYPE
   NAME=c50, JTG, 50000, 70, 0.5, 5, 3
*TDM-LINK
   1, c50,
*ELEM-DEPMATL
   2, NSM, 0.25
*STAGE
   NAME=浇筑, 100, YES, NO, NO, 5
   AELEM=梁, 7
   ABNDR=支座, DEFORMED
   ALOAD=自重, FIRST
*STAGE-CTRL
  YES, , INTERNAL, NO
  0
  NO
  YES, YES, BOTH, 5, 0.01, YES, NO, YES, YES, 0, NO
*ENDDATA
"""


def _convert(text, tmp_path, name="c"):
    p = tmp_path / f"{name}.mct"
    p.write_bytes(text.encode("gbk"))
    model = midas.read(str(p))
    report = midas_case.convert(model, str(tmp_path), name=name)
    return model, report, yaml.safe_load((tmp_path / f"{name}.yaml").read_text(encoding="utf-8"))


def test_the_file_is_read(tmp_path):
    p = tmp_path / "r.mct"; p.write_bytes(MCT.encode("gbk"))
    m = midas.read(str(p))
    assert m.time_models["c50"] == ("JTG", ["50000", "70", "0.5", "5", "3"])
    assert m.time_links[1] == ("c50", "")
    assert m.notional_sizes == {2: 0.25}
    assert m.time_effects == (True, "BOTH", True)


def test_jtg_creep_is_mc90_at_the_mean_strength_of_the_cube_strength(tmp_path):
    _m, report, case = _convert(MCT, tmp_path)
    mats = {k: v for k, v in case["materials"].items() if "creep" in v}
    assert len(mats) == 2                                    # the file's size, and element 2's own
    sizes = sorted(v["creep"]["notional_size"] for v in mats.values())
    assert sizes == [250.0, 500.0]
    for v in mats.values():
        c, s = v["creep"], v["shrinkage"]
        assert c["model"] == "mc90" and c["fcm"] == pytest.approx((0.8 * 50.0 + 8.0) * 1e6)
        assert c["rh"] == 70 and c["cement"] == 5 and c["E28"] == pytest.approx(E * 1e3)
        assert s["drying_start"] == 3 and s["notional_size"] == c["notional_size"]
    st = case["steps"]
    assert case["solver"]["age"] == 7.0                       # the clock starts at the first concrete's age
    assert st[-1]["to_age"] == pytest.approx(107.0)
    assert any("creep and shrinkage" in n for n in report["notes"])


def test_a_cantilever_creeps_by_phi(tmp_path):
    """Determinate: under its weight from age 7, the tip goes down by
    1 + phi(107, 7) of the elastic deflection (each member at its own
    notional size: the two phis bracket it)."""
    from vfife.api import run_case
    from vfife.core import creep
    text = MCT.replace("*TDM-TYPE", "*TDM-TYPE").replace("BOTH, 5", "CREEP, 5")
    _m, _r, case = _convert(text, tmp_path)
    assert all("shrinkage" not in v for v in case["materials"].values())
    case["mesh"]["path"] = str(tmp_path / "c.msh"); case["output"]["path"] = str(tmp_path / "c.h5")
    case["output"]["monitors"] = {"w": {"kind": "displacement", "region": "all", "dof": "z"}}
    import copy
    end = {}
    for n in (1, 2):
        c = copy.deepcopy(case)
        c["steps"] = c["steps"][:n]
        c["output"]["path"] = str(tmp_path / f"c{n}.h5")
        out = run_case(c)
        assert out.converged, out.message
        end[n] = float(np.asarray(out.monitors["w"])[-1])
    ratio = end[2] / end[1]
    phis = [float(creep.creep_coefficient({"model": "mc90", "fcm": 48e6, "rh": 70, "notional_size": h,
                                            "cement": 5}, 107.0, 7.0)) for h in (250.0, 500.0)]
    assert min(phis) < ratio - 1.0 < max(phis) * 1.001, (ratio, phis)


def test_left_out_and_not_taken(tmp_path):
    no_ctrl = MCT[:MCT.index("*STAGE-CTRL")] + "*ENDDATA\n"
    _m, report, case = _convert(no_ctrl, tmp_path, "n")
    assert not any("creep" in v for v in case["materials"].values())
    assert any("no stage control" in n for n in report["notes"])
    off = MCT.replace("YES, YES, BOTH", "NO, YES, BOTH")
    _m, report, case = _convert(off, tmp_path, "o")
    assert not any("creep" in v for v in case["materials"].values())
    aci = MCT.replace("NAME=c50, JTG, 50000, 70, 0.5, 5, 3", "NAME=c50, ACI, 50000, 70, 0.5, 5, 3")
    _m, report, case = _convert(aci, tmp_path, "a")
    assert report["left_out"].get("creep and shrinkage by code ACI (not followed)") == 1


def test_jtg_beta_sc_is_the_shrinkages_alone(tmp_path):
    _m, _r, case = _convert(MCT.replace("0.5, 5, 3", "0.5, 8, 3"), tmp_path, "b")
    m = case["materials"]["m1"]
    assert m["creep"]["cement"] == 5 and m["shrinkage"]["cement"] == 8


def test_the_modulus_grows_where_the_stage_control_says(tmp_path):
    text = MCT.replace("*TDM-LINK\n   1, c50,", "*TDM-ELAST\n   NAME=fc, CODE, CEB-FIP(2010), 50000, 1, 0\n*TDM-LINK\n   1, c50, fc")
    _m, _r, case = _convert(text, tmp_path, "g")
    m = case["materials"]["m1"]
    assert m["ageing"] == {"s": 0.25, "of": "strength"} and "E28" not in m["creep"]
    # bVAR off: the 28-day modulus throughout
    _m, _r, case = _convert(text.replace("0.01, YES, NO, YES", "0.01, YES, NO, NO"), tmp_path, "g2")
    assert "ageing" not in case["materials"]["m1"]


def test_concrete_of_another_age_is_built_just_after_the_stage(tmp_path):
    """Two groups at 7 and 28 days in one stage: the younger with the
    stage's supports and loads, the older in a step of its own right after
    (no time between), with the stage's own constraints; then the time."""
    text = MCT.replace("   梁      , 1to3, 1to2, 0", "   梁      , 1to2, 1, 0\n   梁2      , 2to3, 2, 0") \
              .replace("   AELEM=梁, 7", "   AELEM=梁, 7, 梁2, 28")
    _m, _r, case = _convert(text, tmp_path, "s")
    names = [st["name"] for st in case["steps"]]
    assert names == ["浇筑", "浇筑 (built at 28 d)", "浇筑 (100 d)"], names
    extra = case["steps"][1]
    assert extra["activate_age"] == 28.0 and "to_age" not in extra
    assert case["solver"]["age"] == 7.0
