"""Final review: the documentation as specification.

One test per code item the review found by reading the manuals, the result
format and the references as a user and a third-party tool would:

  M1  a material parameter under two of its names, different values: refused
  M3  defaults that carry a unit are written in the declared units
  M4  a fibre monitor's point is checked against the section, and says what it read
  minors: the refusal's code on the command line, the error-code catalogue,
  `verify` finding its contract beside the case, `api.UserError` and
  `api.verify`'s argument order, `stress_zx`, the six-dof rigid-body advice,
  `describe`/reference fields and modes, few-point plots, `info` units.
"""
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "benchmarks", "assets")


def _cli(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "-m", "vfife", *args], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(path)


# the manual's chapter-3 girder, shortened: one dead-load step and one live
# increment, so that a run takes a few seconds
PC_BEAM = """\
version: 1
name: pc_beam
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {mesh}}}
materials:
  concrete: {{type: concrete02, fpc: 40.0e+6, epsc0: 0.002, fcu: 8.0e+6, epscu: 0.0035,
             ft: 2.6e+6, Ets: 1.5e+9, rat: 0.1, rho: 2500{extra}}}
  rebar:    {{type: steel02, E: 2.0e+11, Fy: 4.0e+8, b: 0.01, R0: 20.0, rho: 7850}}
  strand:   {{type: steel02, E: 1.95e+11, Fy: 1.58e+9, b: 0.02, R0: 20.0, rho: 7850}}
sections:
  phi16:  {{type: bar, material: rebar, shape: circle, diameter: 0.016}}
  duct:   {{type: bar, material: strand, shape: general, area: 2.8e-3}}
  girder:
    type: fiber
    material: concrete
    shape: rect
    width: 0.5
    height: 1.2
    ny: 4
    nz: 24
    integration: lobatto3
    bars:
      - {{y: 0.0, z: -0.55, section: phi16, count: 4}}
    tendons:
      - section: duct
        profile: [[0.0, 0.0, 0.0], [10.0, 0.0, -0.45], [20.0, 0.0, 0.0]]
        interpolation: quadratic
        jack: 1.30e+9
        friction: {{mu: 0.17, k: 0.0015, jack: start, anchor_set: 0.006}}
assign:
  - {{region: beam, section: girder, orientation: [0, 1, 0]}}
constraints:
  - {{region: beam,  dofs: [0, 1, 0, 1, 0, 1]}}
  - {{region: left,  dofs: [1, 1, 1, 0, 0, 0]}}
  - {{region: right, dofs: [0, 1, 1, 0, 0, 0]}}
loads:
  - {{region: beam, line_load: [0.0, 0.0, -15.0e+3]}}
steps:
  - name: dead
    load_steps: 1
solver:
  backend: numpy
  mode: relaxation
  damping: {{type: kinetic}}
  tolerance: 1.0e-5
  max_steps_per_load: 60000
output:
  path: {out}
  fields: [displacement, rotation]
  monitors:
    w_mid: {{kind: displacement, region: mid, dof: z}}
{monitors}"""


def _pc_beam(tmp_path, extra="", monitors="", name="pc_beam.yaml"):
    mesh = shutil.copy(os.path.join(ASSETS, "thermal_span_20.msh"), tmp_path / "thermal_span_20.msh")
    return _write(tmp_path / name, PC_BEAM.format(mesh=os.path.basename(mesh), extra=extra,
                                                  out=str(tmp_path / "pc.h5"), monitors=monitors))


# ---------------------------------------------------------------------- M1
def test_m1_a_parameter_under_two_names_with_two_values_is_refused(tmp_path):
    from vfife import api
    from vfife.materials import base
    with pytest.raises(api.UserError) as exc:
        base.create("concrete02", fpc=40e6, fc=50e6)
    assert exc.value.code == "MATERIAL_ALIAS_CONFLICT" and "`fc`" in exc.value.message
    with pytest.raises(api.UserError, match="`E0`"):
        base.create("steel02", E=2e11, E0=1e11)
    with pytest.raises(api.UserError, match="yield_strength"):
        base.create("steel02", Fy=4e8, yield_strength=5e8)
    with pytest.raises(api.UserError, match="fcu2"):
        base.create("concrete02", fpc=40e6, fcu=8e6, fcu2=6e6)
    with pytest.raises(api.UserError, match="`fc`"):
        base.create("concrete01", fpc=40e6, fc=30e6)
    # the same value under both names, or an alias alone, is read as before
    assert base.create("concrete02", fpc=40e6, fc=40e6).fc_input == 40e6
    assert base.create("concrete02", fc=50e6).fc_input == 50e6
    # the manual's study: refused before the first run, not three equal answers
    case = _pc_beam(tmp_path)
    with pytest.raises(api.UserError, match="name the same parameter") as exc:
        api.study(case, {"materials.concrete.fc": [3e7, 5e7]}, output_dir=str(tmp_path / "s"))
    assert exc.value.code == "STUDY_UNKNOWN_FIELD"
    assert not (tmp_path / "s" / "pc_beam_000.h5").exists()
    # ...and the manual's own example now reads a field the case has
    draft = api.load_case(case)
    assert draft.get("materials.concrete.fpc") == 40e6
    assert draft.get("loads[0].line_load[2]") == -15e3
    manual = open(os.path.join(ROOT, "docs", "user_manual.md"), encoding="utf-8").read()
    assert "materials.concrete.fc\"" not in manual and "materials.concrete.fpc\"" in manual
    assert "materials.concrete.fc\"" not in (api.CaseDraft.__doc__ or "")


# ---------------------------------------------------------------------- M3
def test_m3_unit_carrying_defaults_follow_the_declared_units():
    from vfife.core.units import fill_defaults
    case = {
        "units": {"length": "mm", "force": "N", "stress": "MPa"},
        "materials": {"c": {"type": "concrete02"}, "s": {"type": "steel02", "E0": 2.1e5},
                      "e": {"type": "linear_elastic", "E": 2e5, "nu": 0.3, "rho": 7.85e-9}},
        "sections": {"f": {"type": "fiber", "shear": {"stiffness": 1.0, "stirrups":
                                                     {"area": 1.0, "spacing": 1.0, "depth": 1.0}}}},
        "hydrodynamic_mass": [{"region": "p", "diameter": 2000.0}],
        "loads": [{"region": "d", "vehicle": {"speed": 1.0, "body": {"mass": 1.0},
                                              "axles": [{"offset": 0.0, "suspension": {"stiffness": 1.0}}],
                                              "roughness": {"psd": {"g0": 1.0}}}}],
    }
    out, filled = fill_defaults(case)
    assert out is not case and "rho" not in case["materials"]["c"]            # a copy
    assert out["loads"][0]["vehicle"]["gravity"] == pytest.approx(9806.65)
    assert out["hydrodynamic_mass"][0]["density"] == pytest.approx(1.025e-9)
    assert out["sections"]["f"]["shear"]["stirrups"]["E"] == pytest.approx(2e5)
    psd = out["loads"][0]["vehicle"]["roughness"]["psd"]
    assert (psd["n0"], psd["n_min"], psd["n_max"]) == pytest.approx((1e-4, 1e-5, 3e-3))
    c = out["materials"]["c"]
    assert c["rho"] == pytest.approx(2.4e-9) and c["fpc"] == pytest.approx(30.0)
    s = out["materials"]["s"]
    assert "E" not in s and s["Fy"] == pytest.approx(400.0) and s["rho"] == pytest.approx(7.85e-9)
    assert out["materials"]["e"] == case["materials"]["e"]                     # written: untouched
    assert {w for w, _v, _d in filled} >= {"loads[0].vehicle.gravity", "hydrodynamic_mass[0].density"}
    # SI, or no units: the defaults are SI and the dict is the same object
    for units in ({"length": "m", "force": "N", "stress": "Pa"}, None):
        si = dict(case, units=units)
        assert fill_defaults(si)[0] is si
    # a key left out is SI (here the force, N); a unit it does not know fixes
    # nothing: the default stays SI, and preflight says so
    out, filled = fill_defaults(dict(case, units={"length": "mm"}))
    assert out["hydrodynamic_mass"][0]["density"] == pytest.approx(1.025e-9)
    out, filled = fill_defaults(dict(case, units={"length": "mm", "force": "tonf"}))
    assert out["loads"][0]["vehicle"]["gravity"] == pytest.approx(9806.65)
    assert ("hydrodynamic_mass[0].density", None, "density") in filled


def _vehicle_case(tmp_path, mm):
    src = open(os.path.join(ROOT, "cases", "case_38_sprung_mass_crossing.yaml"), encoding="utf-8").read()
    src = src.replace("../benchmarks/assets/vbi_beam_20.msh", os.path.join(ASSETS, "vbi_beam_20.msh"))
    src = src.replace("total_time: 0.9", "total_time: 0.03")
    src = src.replace("results/case_38_sprung_mass_crossing.h5", str(tmp_path / ("mm.h5" if mm else "m.h5")))
    if mm:
        for a, b in [("units: {length: m, force: N, stress: Pa}", "units: {length: mm, force: N, stress: MPa}"),
                     ("vbi_beam_20.msh}", "vbi_beam_20.msh, scale: 1000.0}"),
                     ("E: 2.87e+9, nu: 0.2, rho: 2303.0", "E: 2870.0, nu: 0.2, rho: 2.303e-9"),
                     ("area: 1.0, iy: 2.9, iz: 2.9, j: 5.8", "area: 1.0e+6, iy: 2.9e+12, iz: 2.9e+12, j: 5.8e+12"),
                     ("speed: 27.78", "speed: 27780.0"), ("mass: 5750.0", "mass: 5.75"),
                     ("stiffness: 1.595e+6", "stiffness: 1595.0")]:
            assert a in src, a
            src = src.replace(a, b)
    p = tmp_path / ("mm.yaml" if mm else "m.yaml")
    p.write_text(src, encoding="utf-8")
    return str(p)


def test_m3_a_vehicle_in_millimetres_weighs_what_it_weighs_in_metres(tmp_path):
    """case_38 in mm / N / MPa / t: 56.3 N of contact force against 56 306 N
    before v1.0, and the deflection a thousand times small."""
    from vfife import api
    si = api.run_case(_vehicle_case(tmp_path, False))
    mm = api.run_case(_vehicle_case(tmp_path, True))
    assert si.converged and mm.converged
    assert mm.monitors["car_force"][-1] == pytest.approx(si.monitors["car_force"][-1], rel=1e-4)
    assert mm.monitors["midspan"][-1] == pytest.approx(1000.0 * si.monitors["midspan"][-1], rel=1e-4)
    codes = [i["code"] for i in api.prepare_case(_vehicle_case(tmp_path, True)).preflight["issues"]]
    assert "UNIT_DEFAULT_CONVERTED" in codes


# ---------------------------------------------------------------------- M4
def test_m4_a_fibre_point_outside_the_section_is_refused(tmp_path):
    from vfife import api
    case = _pc_beam(tmp_path, monitors="    eps: {kind: fiber, region: mid, field: strain, at: [-0.6, 0.0]}\n")
    with pytest.raises(api.UserError) as exc:
        api.run_case(case)
    assert exc.value.code == "FIBER_MONITOR_OUTSIDE"
    assert "[y, z]" in exc.value.message and "z -0.575..0.575" in exc.value.message
    code, text = _cli("check", case, cwd=str(tmp_path))
    assert code == 2 and "FIBER_MONITOR_OUTSIDE" in text
    # fiber_kind belongs to a stress reading
    with pytest.raises(Exception, match="fiber_kind"):
        api.prepare_case(_pc_beam(tmp_path, name="k.yaml", monitors=(
            "    e: {kind: fiber, region: mid, field: strain, at: [0.0, -0.6], fiber_kind: steel}\n")))


def test_m4_a_fibre_monitor_says_which_fibre_it_read(tmp_path):
    import h5py
    from vfife import api
    case = _pc_beam(tmp_path, monitors=(
        "    eps: {kind: fiber, region: mid, field: strain, at: [0.0, -0.6]}\n"
        "    sc: {kind: fiber, region: mid, field: stress, at: [0.0, -0.6], fiber_kind: concrete}\n"
        "    ss: {kind: fiber, region: mid, field: stress, at: [0.0, -0.6], fiber_kind: steel}\n"))
    out = api.run_case(case)
    assert out.converged
    with h5py.File(out.result_path, "r") as f:
        eps, sc, ss = (dict(f["monitors"][k].attrs) for k in ("eps", "sc", "ss"))
    assert eps["read_z"] == -0.6 and eps["read_fiber"].startswith("plane section")
    assert sc["read_fiber"] == "concrete" and sc["read_material"] == "concrete02"
    assert sc["read_z"] == pytest.approx(-0.575)
    assert ss["read_fiber"] == "bar" and ss["read_material"] == "steel02"
    assert ss["read_z"] == pytest.approx(-0.55) and ss["read_station"] in (0.0, 1.0)
    # a bar carries steel's stress, the concrete beside it far less
    assert abs(out.monitors["ss"][-1]) > 5 * abs(out.monitors["sc"][-1])


# -------------------------------------------------------------------- minors
def test_the_command_line_prints_a_refusals_code(tmp_path):
    case = _pc_beam(tmp_path, extra=", fc: 5.0e+7")
    code, text = _cli("run", case, cwd=str(tmp_path))
    assert code == 1 and "error [MATERIAL_ALIAS_CONFLICT]:" in text
    code, text = _cli("check", case, cwd=str(tmp_path))
    assert code == 1 and "error [MATERIAL_ALIAS_CONFLICT]:" in text      # refused as it is read
    # a refusal met while `check` builds the model is reported with its code
    outside = _pc_beam(tmp_path, name="o.yaml", monitors=(
        "    e: {kind: fiber, region: mid, field: strain, at: [-0.6, 0.0]}\n"))
    code, text = _cli("check", outside, cwd=str(tmp_path))
    assert code == 2 and "MODEL_BUILD_FAILED" in text and "(FIBER_MONITOR_OUTSIDE)" in text


def test_the_error_code_catalogue_is_generated_and_complete():
    import re
    from vfife import reference
    text = reference.generate_codes()
    current = open(os.path.join(ROOT, "docs", "error_codes.md"), encoding="utf-8").read()
    assert current == text, "docs/error_codes.md is stale: run `python -m vfife reference --codes`"
    listed = set(re.findall(r"^\| `([A-Z0-9_]+)`", text, re.M))
    for code in ("MATERIAL_ALIAS_CONFLICT", "FIBER_MONITOR_OUTSIDE", "UNIT_DEFAULT_CONVERTED",
                 "RIGID_BODY_MODE", "CHECKPOINT_MISMATCH", "RESTART_OVERWRITES", "MODEL_BUILD_FAILED",
                 "UNKNOWN_OUTPUT_FIELD", "RESULT_TOO_NEW", "VERIFY_ARGUMENTS", "NO_CONTRACT"):
        assert code in listed, code
    # every code the manuals and the result format cite is in it
    statuses = {"API_VERSION", "COMPLETED_DYNAMIC", "MAX_STEPS"}
    for doc in ("user_manual.md", "result_format.md", "diagnostics.md"):
        cited = set(re.findall(r"`([A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)`",
                               open(os.path.join(ROOT, "docs", doc), encoding="utf-8").read()))
        assert cited - listed - statuses - {"PREFLIGHT_FAILED"} == set(), doc


def test_verify_finds_the_contract_beside_the_case(tmp_path):
    cases, ver, work = tmp_path / "proj" / "cases", tmp_path / "proj" / "verification", tmp_path / "work"
    for d in (cases, ver, work):
        d.mkdir(parents=True)
    src = open(os.path.join(ROOT, "cases", "case_02_truss_bar.yaml"), encoding="utf-8").read()
    src = src.replace("../benchmarks/assets/", ASSETS + "/")
    (cases / "case_02_truss_bar.yaml").write_text(src, encoding="utf-8")
    shutil.copy(os.path.join(ROOT, "verification", "case_02_truss_bar.yaml"), ver)
    code, text = _cli("run", str(cases / "case_02_truss_bar.yaml"), cwd=str(work))
    assert code == 0, text
    code, text = _cli("verify", str(cases / "case_02_truss_bar.yaml"), cwd=str(work))
    assert code == 0 and "PASS" in text, text
    code, text = _cli("verify", str(work / "nothing.yaml"), cwd=str(work))
    assert code == 1 and "error [NO_CONTRACT]" in text


def test_user_error_is_api_and_verify_refuses_swapped_arguments(tmp_path):
    from vfife import api
    from vfife.core.diagnostics import UserError
    assert "UserError" in api.__all__ and api.UserError is UserError
    out = api.run_case(os.path.join(ROOT, "cases", "case_02_truss_bar.yaml"), output=str(tmp_path / "t.h5"))
    contract = os.path.join(ROOT, "verification", "case_02_truss_bar.yaml")
    assert api.verify(contract, out.result_path)["status"] == "PASS"
    with pytest.raises(api.UserError) as exc:
        api.verify(out.result_path, contract)
    assert exc.value.code == "VERIFY_ARGUMENTS"
    assert "UserError" in (api.__doc__ or "") and "ValidationError" in api.__doc__


def test_the_sixth_stress_component_is_zx_everywhere():
    from vfife.io.schema import MonitorSpec
    from vfife.verification import _COMPONENT
    from vfife.core.solver import VOIGT_INDEX
    for name in ("zx", "xz"):
        assert MonitorSpec(kind="element", field="stress", component=name).component == name
        assert _COMPONENT[name] == 5 and VOIGT_INDEX[name] == 5
    text = open(os.path.join(ROOT, "docs", "result_format.md"), encoding="utf-8").read()
    assert "`stress_zx`" in text


def test_rigid_body_advice_on_a_six_dof_model_holds_the_rotation(tmp_path):
    from vfife import api
    case = _write(tmp_path / "free.yaml", f"""\
        version: 1
        name: free
        units: {{length: m, force: N, stress: Pa}}
        mesh: {{type: gmsh, path: {os.path.join(ASSETS, 'beam_cantilever_v22.msh')}}}
        materials:
          steel: {{type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850.0}}
        sections:
          beam: {{type: beam, material: steel, shape: general, area: 0.01, iy: 1.0e-4, iz: 1.0e-4, j: 2.0e-4}}
        assign:
        - {{region: all, section: beam, orientation: [0, 0, 1]}}
        constraints:
        - {{region: fixed, dofs: [1, 1, 1, 0, 0, 0]}}
        loads:
        - {{region: loaded, vector: [0, 0, -1000.0]}}
        solver: {{mode: relaxation, n_steps: 600}}
        output: {{path: {tmp_path / 'f.h5'}}}
        """)
    issue = next(i for i in api.prepare_case(case).preflight["issues"] if i["code"] == "RIGID_BODY_MODE")
    lines = [l.strip() for l in issue["message"].splitlines() if l.strip().startswith("free: rotation about x")]
    assert lines and lines[0].endswith("dofs: [0, 0, 0, 1, 0, 0]"), issue["message"]


def test_describe_and_the_reference_name_every_line_field_and_static():
    from vfife.catalog import library
    lib = library()["elements"]
    for el in ("beam2", "beam2_timoshenko", "mixed_line", "beam2_fiber"):
        fields = set(lib[el]["output_fields"])
        assert {"axial_force", "bending_y", "bending_z_max_abs", "member_failed"} <= fields, el
    assert {"axial_force_max_abs", "von_mises", "capacity_ratio"} <= set(lib["truss2"]["output_fields"])
    ref = open(os.path.join(ROOT, "docs", "yaml_reference.md"), encoding="utf-8").read()
    row = next(l for l in ref.splitlines() if l.startswith("| `beam2` |"))
    assert "static" in row and "bending_y_max_abs" in row
    code, text = _cli("describe", "beam2")
    assert "static" in text and "axial_force_max_abs" in text


def test_a_one_record_series_is_drawn_with_markers(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from vfife import api
    from vfife.post import plot
    out = api.run_case(os.path.join(ROOT, "cases", "case_02_truss_bar.yaml"), output=str(tmp_path / "t.h5"))
    seen = []
    real = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: (seen.append(self), real(self, *a, **k))[1])
    plot(out.result_path, str(tmp_path / "a.png"), x="loaded_ux", y=["support_fx"])
    plot(out.result_path, str(tmp_path / "b.png"))
    markers = [ln.get_marker() for fig in seen for ax in fig.axes for ln in ax.get_lines()
               if len(ln.get_xdata()) == 1]
    assert markers and all(m == "o" for m in markers)


def test_info_prints_mass_and_weight_in_the_declared_units(tmp_path):
    case = _pc_beam(tmp_path)
    code, text = _cli("info", case, cwd=str(tmp_path))
    assert code == 0 and "kg" in text and "N at g = 9.80665" in text
    mm = _vehicle_case(tmp_path, True)
    code, text = _cli("info", mm, cwd=str(tmp_path))
    assert code == 0, text
