"""Settings that validated and were then ignored, as a targeted review of the
older code found them. One test per reproduced defect: each
is now refused by name (or warned about, where it is legal but surprising),
or honoured.
"""
import copy

import h5py
import numpy as np
import pytest

from vfife import api
from vfife.core.diagnostics import UserError
from test_member_failure import _write_msh

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}
CONC = {"type": "concrete01", "fc": -3.0e7, "eps_c0": -0.002, "fcu": -6.0e6, "eps_cu": -0.0035, "rho": 2500.0}
BAR = {"type": "steel02", "E": 2.0e11, "Fy": 4.0e8, "b": 0.01, "rho": 7850.0}


def _cantilever(tmp_path, n=4, L=2.0):
    _write_msh(tmp_path / "c.msh", [(L * i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]),
                "bar": (1, [(i, i + 1) for i in range(n)]),
                "half": (1, [(i, i + 1) for i in range(n // 2)])})
    return {"version": 1, "name": "c", "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": "c.msh"},
            "materials": {"steel": dict(STEEL), "conc": dict(CONC), "bar": dict(BAR)},
            "sections": {"beam": {"type": "beam", "material": "steel", "shape": "rect",
                                  "width": 0.1, "height": 0.1},
                         "fib": {"type": "fiber", "material": "conc", "shape": "rect",
                                 "width": 0.3, "height": 0.4, "ny": 1, "nz": 8},
                         "rebar": {"type": "bar", "material": "bar", "shape": "general", "area": 3e-4}},
            "assign": [{"region": "bar", "section": "beam", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "root", "dofs": [1] * 6}],
            "loads": [{"region": "tip", "vector": [0, 0, -1.0e3]}],
            "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 4000},
            "output": {"fields": ["displacement"],
                       "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}}}


def _prepare(tmp_path, d):
    return api.prepare_case(api.CaseDraft(d, base=tmp_path))


def _codes(tmp_path, d, severity=("ERROR",)):
    rep = _prepare(tmp_path, d).preflight
    return {i["code"] for i in rep["issues"] if i["severity"] in severity}


def _refused(tmp_path, d, match):
    with pytest.raises((ValueError, UserError), match=match):
        _prepare(tmp_path, d)


# ------------------------------------------------------------ the input form
def test_a_misspelt_section_name_is_a_sentence_not_a_traceback(tmp_path):
    d = _cantilever(tmp_path)
    d["assign"][0]["section"] = "Beam"
    _refused(tmp_path, d, r"section 'Beam', which is not in `sections` \(beam, fib, rebar\)")


def test_a_fibre_bar_without_z_is_refused(tmp_path):
    d = _cantilever(tmp_path)
    d["sections"]["fib"]["bars"] = [{"y": 0.0, "section": "rebar"}]
    d["assign"][0]["section"] = "fib"
    _refused(tmp_path, d, "bars\\[0\\] needs numeric `y` and `z`")


@pytest.mark.parametrize("section, key, value", [
    ("fib", "releases", ["i_ry"]),                           # the fibre element has no releases
    ("beam", "prestress", 1.0e9),                            # a beam carries no prestress
    ("beam", "tension_only", True),
])
def test_an_assignment_key_the_section_does_not_read_is_refused(tmp_path, section, key, value):
    d = _cantilever(tmp_path)
    d["assign"][0].update(section=section, **{key: value})
    _refused(tmp_path, d, f"does not read `{key}`")


@pytest.mark.parametrize("section, key, value", [
    ("beam", "tendons", [{"profile": [[0, 0, -0.03], [1, 0, -0.03]], "section": "rebar"}]),
    ("beam", "shear_stiffness", 1.0e9),
    ("fib", "theory", "timoshenko"),
])
def test_a_section_key_meant_for_another_type_is_refused(tmp_path, section, key, value):
    d = _cantilever(tmp_path)
    d["sections"][section][key] = value
    d["assign"][0]["section"] = section
    _refused(tmp_path, d, f"does not read `{key}`")


def test_a_core_law_without_a_cover_is_refused(tmp_path):
    d = _cantilever(tmp_path)
    d["sections"]["fib"]["core_material"] = "conc"
    d["assign"][0]["section"] = "fib"
    _refused(tmp_path, d, "core_material")


def test_a_bar_outside_the_section_is_refused(tmp_path):
    d = _cantilever(tmp_path)
    d["sections"]["fib"]["bars"] = [{"y": 0.0, "z": 0.9, "section": "rebar"}]
    d["assign"][0]["section"] = "fib"
    _refused(tmp_path, d, "outside the section")


def test_a_single_fibre_section_on_part_of_the_mesh_is_refused(tmp_path):
    """It was applied to every member whatever its region said."""
    d = _cantilever(tmp_path)
    d["assign"][0].update(section="fib", region="half")
    with pytest.raises((ValueError, UserError), match="region 'half', which owns 2 of the 4"):
        api.run_case(api.CaseDraft(d, base=tmp_path), output=str(tmp_path / "h.h5"))


def test_a_solid_section_on_a_line_mesh_is_refused_in_section_terms(tmp_path):
    d = _cantilever(tmp_path)
    d["sections"] = {"body": {"type": "solid", "material": "steel"}}
    d["assign"] = [{"region": "bar", "section": "body"}]
    _refused(tmp_path, d, "sections are solid or shell, but c.msh contains only Line2")


def test_a_shell_needs_only_material_and_thickness():
    """Manual 7.1; the lowering copied a formulation the shell refuses."""
    from vfife.io import model, schema
    d = {"version": 1, "name": "s", "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": "s.msh"}, "materials": {"steel": dict(STEEL)},
         "sections": {"plate": {"type": "shell", "material": "steel", "thickness": 0.01}},
         "assign": [{"region": "all", "section": "plate"}],
         "constraints": [], "loads": [], "solver": {"mode": "relaxation", "n_steps": 10}}
    low = model.lower(d)
    assert low["element"]["formulation"] == "selective"
    schema.Case(**low)                                 # was: shell supports selective only


# --------------------------------------------------------------- preflight
def test_monitors_a_frame_cannot_record_are_refused(tmp_path):
    d = _cantilever(tmp_path)
    d["output"]["monitors"] = {"s": {"kind": "element", "region": "bar", "field": "von_mises"},
                               "g": {"kind": "fiber", "region": "bar", "field": "strain", "at": [0.05, 0.0]},
                               "r": {"kind": "displacement", "region": "tip", "dof": "z", "reduce": "min"}}
    codes = _codes(tmp_path, d)
    assert {"MONITOR_UNSUPPORTED", "MONITOR_REDUCE_UNUSED"} <= codes


@pytest.mark.parametrize("change, code", [
    (lambda d: d["solver"].update(erosion={"damage": 0.01}), "EROSION_UNSUPPORTED"),
    (lambda d: d["loads"][0].update(follower=True, vector=None, moment=[0, 1e3, 0]) or d["loads"][0].pop("vector"),
     "FOLLOWER_UNSUPPORTED"),
])
def test_settings_a_frame_does_not_use_are_refused(tmp_path, change, code):
    d = _cantilever(tmp_path)
    change(d)
    assert code in _codes(tmp_path, d)


def test_a_steel02_cable_is_refused_whichever_assignment_comes_first(tmp_path):
    _write_msh(tmp_path / "t.msh", [(0, 0, 0), (1, 0, 0), (1, 0, -1)],
               {"root": (0, [0]), "tip": (0, [1]), "w": (0, [2]),
                "arm": (1, [(0, 1)]), "cab": (1, [(1, 2)])})
    base = _cantilever(tmp_path)
    base["mesh"]["path"] = "t.msh"
    base["sections"]["cable"] = {"type": "bar", "material": "bar", "area": 1e-4}
    beam = {"region": "arm", "section": "beam", "orientation": [0, 1, 0]}
    cable = {"region": "cab", "section": "cable"}
    base["loads"] = [{"region": "w", "vector": [0, 0, -1e3]}]
    base["output"]["monitors"] = {}
    for order in ([beam, cable], [cable, beam]):
        d = copy.deepcopy(base)
        d["assign"] = copy.deepcopy(order)
        assert "CAPABILITY_MISMATCH" in _codes(tmp_path, d)


def test_amplitudes_are_checked_before_the_run(tmp_path):
    d = _cantilever(tmp_path)
    d["loads"][0]["amplitude"] = "Hold"
    d["amplitudes"] = {"hold": [[0, 1], [1, 1]], "secs": [[0, 0], [5, 1], [10, 1]]}
    rep = _prepare(tmp_path, d).preflight
    codes = {i["code"]: i["severity"] for i in rep["issues"]}
    assert codes["AMPLITUDE_UNDEFINED"] == "ERROR"
    assert codes["AMPLITUDE_UNUSED"] == "WARNING"
    assert codes["AMPLITUDE_PAST_PROGRESS"] == "WARNING"


def test_a_model_level_ramp_in_a_two_step_case_is_warned_about(tmp_path):
    d = _cantilever(tmp_path)
    d["amplitudes"] = {"up": [[0, 0], [1, 1]]}
    d["loads"][0]["amplitude"] = "up"
    d["steps"] = [{"name": "a", "load_steps": 2}, {"name": "b", "load_steps": 2}]
    assert "AMPLITUDE_REPEATS" in _codes(tmp_path, d, ("WARNING",))


# ---------------------------------------------------------- what is honoured
def test_a_misspelt_field_is_refused_on_a_beam_and_end_actions_are_written(tmp_path):
    """The probe raised on every line model and the check was skipped; and
    `end_actions`, asked for by thirteen shipped cases, wrote nothing."""
    d = _cantilever(tmp_path)
    d["output"]["fields"] = ["displacement", "von_misses"]
    with pytest.raises(UserError, match="von_misses"):
        api.run_case(api.CaseDraft(d, base=tmp_path), output=str(tmp_path / "x.h5"))
    d["output"]["fields"] = ["displacement", "end_actions"]
    d["solver"].update(mode="static", load_steps=1)
    out = api.run_case(api.CaseDraft(d, base=tmp_path), output=str(tmp_path / "e.h5"))
    assert out.status == "CONVERGED"
    with h5py.File(tmp_path / "e.h5") as f:
        last = f["steps"][sorted(f["steps"])[-1]]
        assert {"bending_y_i", "bending_y_j", "axial_force_i"} <= set(last)
        M = np.asarray(last["bending_y_i"])
    assert abs(M[0]) == pytest.approx(1.0e3 * 2.0, rel=1e-3)          # P L at the root


def test_a_step_tolerance_is_honoured_by_a_frame(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"].update(n_steps=400)
    d["steps"] = [{"name": "tight", "tolerance": 1e-14}]
    out = api.run_case(api.CaseDraft(d, base=tmp_path), output=str(tmp_path / "t.h5"))
    assert out.status == "MAX_STEPS" and "1.000e-14" in out.message


def test_a_misspelt_material_parameter_is_refused(tmp_path):
    d = _cantilever(tmp_path)
    d["materials"]["bar"]["R00"] = 20.0
    d["sections"]["fib"]["bars"] = [{"y": 0.0, "z": 0.1, "section": "rebar"}]
    d["assign"][0]["section"] = "fib"
    with pytest.raises(UserError, match="R00.*R0"):
        api.run_case(api.CaseDraft(d, base=tmp_path), output=str(tmp_path / "m.h5"))
