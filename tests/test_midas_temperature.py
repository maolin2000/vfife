"""Temperatures of a Midas load case (1.23): the whole model's
(`*SYSTEMPER`), an element's (`*ELTEMPER`) and one over a section's depth
(`*BSTEMPER`, reduced to its uniform change and linear gradient), as one
more step through `convert(..., cases=...)`."""
import numpy as np
import pytest
import yaml

from vfife.io import midas, midas_case
from test_midas_creep import MCT as CREEP_MCT

ALPHA, L = 1.0e-5, 10.0                  # the material's, 1/K; the cantilever's length, m


def _mct(block):
    return (CREEP_MCT.replace("YES, YES, BOTH, 5", "NO, YES, BOTH, 5")           # no creep here
            .replace("*TDM-TYPE", f"*STLDCASE\n   温度 , USER,\n*USE-STLD, 温度\n{block}\n*TDM-TYPE"))


SYSTEM = "*SYSTEMPER\n   20, "
ELEMENT = "*ELTEMPER\n   2, 30, "
# the whole depth of the 1 m square, 20 K at the top falling to 0 at the bottom
SECTION = "*BSTEMPER\n   1, LZ, Top, 1, , NO\n   ELEMENT, 0, 0, 1, 0, 20, 1, 0\n" \
          "   2, LZ, Top, 1, , NO\n   ELEMENT, 0, 0, 1, 0, 20, 1, 0"
# two layers in the top 0.5 m: 20 -> 5 K over 0.2 m, 5 -> 0 over 0.3 m, 0.6 m wide
TWO_LAYERS = "*BSTEMPER\n   1to2, LZ, Top, 2, , NO\n   ELEMENT, 0, 0, 0.6, 0, 20, 0.2, 5\n" \
             "   ELEMENT, 0, 0, 0.6, 0.2, 5, 0.5, 0"


def _convert(tmp_path, block, name="t"):
    p = tmp_path / f"{name}.mct"
    p.write_bytes(_mct(block).encode("gbk"))
    model = midas.read(str(p))
    report = midas_case.convert(model, str(tmp_path), name=name, cases=("温度",))
    return model, report, yaml.safe_load((tmp_path / f"{name}.yaml").read_text(encoding="utf-8"))


def test_the_file_is_read(tmp_path):
    p = tmp_path / "r.mct"
    p.write_bytes(_mct(ELEMENT + "\n" + TWO_LAYERS).encode("gbk"))
    m = midas.read(str(p))
    assert m.element_temperatures == [("温度", [2], 30.0, "")]
    assert m.section_temperatures == [["温度", [1, 2], "LZ", "TOP", [("ELEMENT", 0.6, 0.0, 20.0, 0.2, 5.0), ("ELEMENT", 0.6, 0.2, 5.0, 0.5, 0.0)],
                                       "", False]]


def _temperatures(case):
    last = case["steps"][-1]
    assert last["name"] == "温度"
    return [ld for ld in last.get("loads", []) if "temperature" in ld]


def test_two_layers_are_their_uniform_change_and_linear_gradient(tmp_path):
    _m, _r, case = _convert(tmp_path, TWO_LAYERS)
    (ld,) = _temperatures(case)
    # integrate the profile itself: T(h), h below the top, z = 0.5 - h
    h = np.linspace(0.0, 0.5, 50001)
    T = np.interp(h, [0.0, 0.2, 0.5], [20.0, 5.0, 0.0])
    b, z = 0.6, 0.5 - h
    uniform = np.trapezoid(T * b, h) / 1.0
    gradient = np.trapezoid(T * b * z, h) / (1.0 / 12.0)
    assert ld["temperature"]["uniform"] == pytest.approx(uniform, rel=1e-6)
    assert ld["temperature"]["gradient"] == pytest.approx(gradient, rel=1e-6)
    assert ld["temperature"]["axis"] == "z" and ld["keep"]


@pytest.mark.parametrize("block, ux, uz", [
    (SYSTEM, ALPHA * 20 * L, 0.0),                               # the whole model 20 K warmer
    (ELEMENT, ALPHA * 30 * L / 2, 0.0),                          # element 2 alone, 30 K
    # 10 K on average, 20 K/m through the depth: the free cantilever curves
    # by alpha dT/dz, the top longer -- the tip goes down by kappa L^2 / 2
    (SECTION, ALPHA * 10 * L, -ALPHA * 20 * L ** 2 / 2),
], ids=["system", "element", "section"])
def test_a_free_cantilever_takes_the_temperature_without_stress(tmp_path, block, ux, uz):
    from vfife.api import run_case
    _m, report, case = _convert(tmp_path, block)
    assert _temperatures(case)
    case["mesh"]["path"] = str(tmp_path / "t.msh")
    case.setdefault("regions", {})["tip"] = {"where": {"x": L, "y": 0.0, "z": 0.0}}
    case["output"]["monitors"] = {d_: {"kind": "displacement", "region": "tip", "dof": d_} for d_ in ("x", "z")}
    got = []
    for n in (len(case["steps"]) - 1, len(case["steps"])):
        cc = dict(case, steps=case["steps"][:n])
        cc["output"] = dict(case["output"], path=str(tmp_path / f"t{n}.h5"))
        out = run_case(cc)
        assert out.converged, out.message
        got.append(np.array([float(np.asarray(out.monitors[d_])[-1]) for d_ in ("x", "z")]))
    d = got[1] - got[0]
    # to within what the cantilever's sag under its own weight (11 mm, a
    # slope of 1.5e-3 at the tip) turns of one displacement into the other
    assert d[0] == pytest.approx(ux, abs=3e-5)
    assert d[1] == pytest.approx(uz, abs=3e-5)


def test_what_is_not_followed_is_counted(tmp_path):
    block = "*BSTEMPER\n   1, LY, Top, 1, , NO\n   ELEMENT, 0, 0, 1, 0, 20, 1, 0\n" \
            "   2, LZ, Top, 1, , YES\n   ELEMENT, 0, 0, 1, 0, 20, 1, 0"
    _m, report, case = _convert(tmp_path, block)
    text = str(report)
    assert "across the width" in text
    assert "PSC" in text
    assert not _temperatures(case)


@pytest.mark.parametrize("head, layer, why", [
    ("1, LZ, Cent, 1, , NO", "ELEMENT, 0, 0, 1, 0, 20, 1, 0", "measured from CENT"),
    ("1, LZ, Top, 1, , NO", "INPUT, 2.06e8, 1.2e-5, 1, 0, 20, 1, 0", "a layer of its own material"),
    ("1, LZ, Top, 2, , NO", "ELEMENT, 0, 0, 1, 0, 20", "PSC form, or with a layer not read"),
], ids=["face", "material", "short row"])
def test_a_profile_not_followed_is_counted(tmp_path, head, layer, why):
    _m, report, case = _convert(tmp_path, f"*BSTEMPER\n   {head}\n   {layer}")
    assert any(why in k for k in report["left_out"]), report["left_out"]
    assert not _temperatures(case)


def test_a_grade_named_without_values_takes_the_codes_coefficient(tmp_path):
    p = tmp_path / "g.mct"
    p.write_bytes(_mct(SYSTEM).replace(
        "1, CONC, C50 , 0, 0, , C, NO, 0.05, 2,  3.4500e+07,   0.2,  1.0000e-05, 25,     0",
        "1, CONC, C50 , 0, 0, , C, NO, 0.05, 1, JTG04(RC), , C50, NO, 3.45e+07").encode("gbk"))
    m = midas.read(str(p))
    assert m.materials[1].alpha is None and m.materials[1].standard
    report = midas_case.convert(m, str(tmp_path), name="g", cases=("温度",))
    case = yaml.safe_load((tmp_path / "g.yaml").read_text(encoding="utf-8"))
    assert case["materials"]["m1"]["thermal_expansion"] == 1.0e-5
    assert any("thermal coefficient 1e-05/K assumed" in n for n in report["notes"])


def test_a_case_a_stage_already_applies_is_refused(tmp_path):
    with pytest.raises(ValueError, match="already applied by a stage"):
        _convert(tmp_path, "*SYSTEMPER\n   20, 自重")
