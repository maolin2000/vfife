"""Imposed support displacements from a Midas file (1.23): a case's
`*SPDISP` and a settlement group (`*SM-GROUP`), as one more step through
`convert(..., cases=...)`."""
import numpy as np
import pytest
import yaml

from vfife.io import midas, midas_case
from test_midas_creep import MCT as CREEP_MCT

E, L, I = 3.45e10, 10.0, 1.0 / 12.0          # the 1 m square section, Pa, m, m4

MCT = (CREEP_MCT
       .replace("YES, YES, BOTH, 5", "NO, YES, BOTH, 5")                       # no creep here
       .replace("   1, 111111, 支座\n", "   1, 111111, 支座\n   3, 001000, 支座\n")
       .replace("*TDM-TYPE", """*STLDCASE
   沉降 , USER,
*USE-STLD, 沉降
*SPDISP
   3, 001000, 0, 0, -0.01, 0, 0, 0,
*SM-GROUP
   右支座, -0.02, 3
*TDM-TYPE"""))


def _convert(tmp_path, cases):
    p = tmp_path / "s.mct"
    p.write_bytes(MCT.encode("gbk"))
    model = midas.read(str(p))
    report = midas_case.convert(model, str(tmp_path), name="s", cases=cases)
    return model, report, yaml.safe_load((tmp_path / "s.yaml").read_text(encoding="utf-8"))


def test_the_file_is_read(tmp_path):
    p = tmp_path / "r.mct"; p.write_bytes(MCT.encode("gbk"))
    m = midas.read(str(p))
    assert m.displacements == [("沉降", [3], "001000", (0.0, 0.0, -0.01, 0.0, 0.0, 0.0), "")]
    assert m.settlements == {"右支座": (-0.02, [3])}


@pytest.mark.parametrize("case, dz", [("沉降", -0.01), ("右支座", -0.02)])
def test_a_prop_that_settles_loses_3_E_I_delta_over_L_cubed(tmp_path, case, dz):
    from vfife.api import run_case
    _m, report, c = _convert(tmp_path, (case,))
    last = c["steps"][-1]
    assert last["name"] == case
    con = [k for k in last["constraints"] if k.get("value") is not None]
    assert len(con) == 1 and con[0]["value"] == [0.0, 0.0, dz] and con[0]["keep"]
    c["mesh"]["path"] = str(tmp_path / "s.msh"); c["output"]["path"] = str(tmp_path / "s.h5")
    reg = con[0]["region"]
    c["output"]["monitors"] = {"R": {"kind": "reaction", "region": reg, "dof": "z"},
                               "w": {"kind": "displacement", "region": reg, "dof": "z"}}
    got = {}
    for n in (len(c["steps"]) - 1, len(c["steps"])):
        cc = dict(c, steps=c["steps"][:n])
        cc["output"] = dict(c["output"], path=str(tmp_path / f"s{n}.h5"))
        out = run_case(cc)
        assert out.converged, out.message
        got[n] = {k: float(np.asarray(v)[-1]) for k, v in out.monitors.items()}
    n0, n1 = len(c["steps"]) - 1, len(c["steps"])
    assert got[n1]["w"] == pytest.approx(dz, abs=1e-9)
    # the propped cantilever: the prop's reaction falls by 3 E I delta / L^3
    assert got[n1]["R"] - got[n0]["R"] == pytest.approx(3 * E * I * dz / L ** 3, rel=2e-2)


def test_an_unknown_case_names_the_settlement_groups(tmp_path):
    with pytest.raises(ValueError, match="settlement groups: 右支座"):
        _convert(tmp_path, ("没有",))
