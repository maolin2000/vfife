"""The final, scenario-based review: service life to collapse.

A bridge engineer's four workflows end to end -- a post-tensioned girder from
transfer to 30 years, a vehicle and an influence line, a corroded girder
pushed to collapse, a hanger cut and the progressive failure after it. Each
test is one finding: what it did before is in its text.
"""
import copy
import json
import pathlib
import sys

import h5py
import numpy as np
import pytest
import yaml

from vfife import api
from vfife.core.diagnostics import UserError

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPAN = ROOT / "benchmarks" / "assets" / "thermal_span_20.msh"      # 20 m, beam/left/mid/right
sys.path.insert(0, str(ROOT / "tests"))
from test_member_failure import _hanging  # noqa: E402

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}
UNITS = {"length": "m", "force": "N", "stress": "Pa"}


def _run(tmp_path, d, name, **kw):
    p = tmp_path / f"{name}.yaml"
    d = copy.deepcopy(d)
    d["name"] = name
    d.setdefault("output", {})["path"] = str(tmp_path / f"{name}.h5")
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return api.run_case(str(p), **kw), tmp_path / f"{name}.h5", p


def _last(h5, *names):
    with h5py.File(h5) as f:
        g = f["steps"][sorted(f["steps"], key=int)[-1]]
        return [np.asarray(g[n], float) for n in names]


def _strut(fields=None, steps=None, monitors=None, every=None):
    """A 20 m pin-roller beam under 100 kN of axial COMPRESSION and 10 kN at midspan."""
    out = {"fields": fields or ["displacement", "axial_force", "axial_force_max_abs", "bending_y",
                                "bending_y_max_abs", "shear_z", "end_actions"],
           "monitors": monitors or {"w": {"kind": "displacement", "region": "mid", "dof": "z"}}}
    if every is not None:
        out["every"] = every
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": str(SPAN)},
         "materials": {"steel": STEEL},
         "sections": {"s": {"type": "beam", "material": "steel", "shape": "general",
                            "area": 0.1, "iy": 0.1, "iz": 0.1, "j": 0.1}},
         "assign": [{"region": "beam", "section": "s", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "beam", "dofs": [0, 1, 0, 1, 0, 1]},
                         {"region": "left", "dofs": [1, 1, 1, 0, 0, 0]},
                         {"region": "right", "dofs": [0, 1, 1, 0, 0, 0]}],
         "loads": [{"region": "right", "vector": [-1.0e5, 0, 0]},
                   {"region": "mid", "vector": [0, 0, -1.0e4]}],
         "solver": {"mode": "static", "backend": "numpy", "tolerance": 1.0e-6},
         "output": out}
    if steps is not None:
        d["steps"] = steps
    return d


# ---------------------------------------------------------------- M2
def test_member_forces_are_signed_tension_positive(tmp_path):
    """`axial_force` was max(|end i|, |end j|): a strut under 1 MN of
    compression reported +1.0e6, sagging and hogging were both positive."""
    out, h5, _ = _run(tmp_path, _strut(), "strut")
    assert out.converged
    N, Nabs, M, Mabs, V, Ni, Nj = _last(h5, "axial_force", "axial_force_max_abs", "bending_y",
                                          "bending_y_max_abs", "shear_z", "axial_force_i", "axial_force_j")
    assert np.all(N < 0) and np.allclose(N, -1.0e5, rtol=1e-3)          # compression
    np.testing.assert_allclose(Nabs, np.abs(N))                         # the old field
    np.testing.assert_allclose(-Ni, N, rtol=1e-6)                       # internal = -f_i = +f_j
    np.testing.assert_allclose(Nj, N, rtol=1e-6)
    assert np.all(M[8:12] < 0)                                          # sagging, local z up
    np.testing.assert_allclose(Mabs, np.abs(M))
    assert V[0] < 0 < V[-1]                                             # left part pushed down by the right


def _cantilever(section):
    return {"units": UNITS, "mesh": {"type": "gmsh", "path": str(SPAN)},
            "materials": {"steel": STEEL}, "sections": {"s": section},
            "assign": [{"region": "beam", "section": "s", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "left", "dofs": [1, 1, 1, 1, 1, 1]}],
            "loads": [{"region": "right", "vector": [0, 1.0e4, 2.0e4]},
                      {"region": "right", "moment": [3.0e4, 0, 0]}],
            "solver": {"mode": "static", "backend": "numpy", "tolerance": 1.0e-6},
            "output": {"fields": ["displacement", "rotation", "axial_force", "shear_y", "shear_z",
                                  "torsion", "bending_y", "bending_z", "end_actions"]}}


def test_fibre_and_frame_members_share_one_sign_convention(tmp_path):
    """The fibre path reported |N|, |My|, ... at its largest station and
    |GA gamma| for the shear; it is signed now in the frames' convention."""
    names = ("shear_y", "shear_z", "torsion", "bending_y", "bending_z")
    beam = {"type": "beam", "material": "steel", "shape": "rect", "width": 1.0, "height": 1.0, "j": 0.1}
    fibre = {"type": "fiber", "material": "steel", "shape": "rect", "width": 1.0, "height": 1.0,
             "ny": 4, "nz": 4, "shear_stiffness": 1.0e10, "j": 0.1}
    got = {}
    for tag, sec in (("frame", beam), ("fibre", fibre)):
        out, h5, _ = _run(tmp_path, _cantilever(sec), f"cant_{tag}")
        assert out.converged
        got[tag] = dict(zip(names, (a[0] for a in _last(h5, *names))))
    # the root: (40-x) x F for the moments, +F for the shears, +T for the torque
    for tag in got:
        r = got[tag]
        assert r["shear_y"] > 0 and r["shear_z"] > 0 and r["torsion"] > 0
        assert r["bending_y"] < 0 < r["bending_z"]
    for n in names:
        assert got["fibre"][n] == pytest.approx(got["frame"][n], rel=0.03)


def test_the_fibre_kernel_and_the_reference_path_give_the_same_signed_forces(tmp_path):
    from vfife.cli import _with_element, build, make_solver
    from vfife.io import schema
    sec = {"type": "fiber", "material": "steel02", "shape": "rect", "width": 1.0, "height": 1.0,
           "ny": 4, "nz": 4, "shear_stiffness": 1.0e10, "j": 0.1}
    d = _cantilever(sec)
    d["materials"] = {"steel02": {"type": "steel02", "E": 2.0e11, "Fy": 3.45e9, "b": 0.01,
                                  "R0": 20.0, "rho": 7850.0}}
    p = tmp_path / "k.yaml"
    d["name"] = "k"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    case, _ = schema.load(str(p))
    mesh, material, element = build(case, str(p))
    s = make_solver(_with_element(case, element, str(p)), mesh, material)
    if getattr(s, "_kernel", None) is None:
        pytest.skip("no fused kernel here: " + str(getattr(s, "kernel_status", "")))
    rng = np.random.default_rng(3)
    s.q[:] = 1e-4 * rng.standard_normal(s.q.shape)
    s.q[s.dofs.node_dofs(np.array([0]), ("ux", "uy", "uz", "rx", "ry", "rz")).ravel()] = 0.0
    a_kernel = np.array(s.internal_force()[1], copy=True)
    s._kernel = None
    a_numpy = np.array(s.internal_force()[1], copy=True)
    assert np.any(a_kernel < 0)                               # signed
    np.testing.assert_allclose(a_kernel, a_numpy, rtol=1e-7, atol=1e-6 * np.abs(a_numpy).max())


def test_a_bar_chain_is_signed_and_is_not_a_mechanism(tmp_path):
    """A straight bar chain of a mixed_line model, held transversely at every
    node, was refused for "rotation about x" (RIGID_BODY_MODE), with a fix
    that did not fix it; its bars report compression negative."""
    d = _strut(fields=["displacement", "axial_force", "axial_force_max_abs"])
    d["sections"] = {"s": {"type": "bar", "material": "steel", "shape": "general", "area": 0.1}}
    d["assign"] = [{"region": "beam", "section": "s"}]
    d["constraints"] = [{"region": "beam", "dofs": [0, 1, 1, 0, 0, 0]},
                        {"region": "left", "dofs": [1, 1, 1, 0, 0, 0]}]
    d["loads"] = [{"region": "right", "vector": [-1.0e5, 0, 0]}]
    p = tmp_path / "bars.yaml"
    d["name"] = "bars"
    d["output"]["path"] = str(tmp_path / "bars.h5")
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    prep = api.prepare_case(str(p))
    assert prep.preflight["status"] == "PASS", prep.preflight["issues"]
    out = api.run_case(prep)
    assert out.converged
    N, Nabs = _last(tmp_path / "bars.h5", "axial_force", "axial_force_max_abs")
    np.testing.assert_allclose(N, -1.0e5, rtol=1e-6)
    np.testing.assert_allclose(Nabs, 1.0e5, rtol=1e-6)


# ---------------------------------------------------------------- 3
def _creep_beam(sdl_amplitude=None):
    load = {"region": "beam", "line_load": [0.0, 0.0, -1.0e4]}
    if sdl_amplitude:
        load["amplitude"] = sdl_amplitude
    conc = {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2400.0,
            "creep": {"model": "curve", "curve": [[0, 0.0], [10.0, 1.0], [1000.0, 2.0]]}}
    return {"units": UNITS, "mesh": {"type": "gmsh", "path": str(SPAN)},
            "materials": {"c": conc},
            "sections": {"s": {"type": "beam", "material": "c", "shape": "rect", "width": 0.5, "height": 1.0}},
            "assign": [{"region": "beam", "section": "s", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "beam", "dofs": [0, 1, 0, 1, 0, 1]},
                            {"region": "left", "dofs": [1, 1, 1, 0, 0, 0]},
                            {"region": "right", "dofs": [0, 1, 1, 0, 0, 0]}],
            "loads": [{"region": "beam", "line_load": [0.0, 0.0, -2.0e4]}],
            "steps": [{"name": "dead", "age": 28.0, "load_steps": 1},
                      {"name": "sdl", "age": 90.0, "load_steps": 1, "loads": [dict(load)]},
                      {"name": "creep", "to_age": 1000.0, "load_steps": 4, "loads": [dict(load)]}],
            "solver": {"mode": "relaxation", "backend": "numpy", "age": 28.0,
                       "damping": {"type": "kinetic"}, "tolerance": 1.0e-7, "max_steps_per_load": 40000},
            "output": {"monitors": {"R": {"kind": "reaction", "region": "left", "dof": "z"},
                                    "w": {"kind": "displacement", "region": "mid", "dof": "z"}}}}


def test_loads_in_a_time_step_hold_by_default(tmp_path):
    """A superimposed dead load repeated in a `to_age` step took the staged
    default `ramp`: it dropped from full to 1/4 at the first time station and
    climbed back over the years (the 30-year deflection of the scenario
    review: +1.1 mm instead of -3.2 mm)."""
    a, _, _ = _run(tmp_path, _creep_beam(), "default")
    b, _, _ = _run(tmp_path, _creep_beam("hold"), "held")
    assert a.converged and b.converged
    R = np.asarray(a.monitors["R"])
    np.testing.assert_allclose(R[2:], 3.0e5, rtol=1e-4)       # (20 + 10) kN/m x 20 m / 2, at every station
    np.testing.assert_allclose(a.monitors["w"], b.monitors["w"], rtol=1e-9)


# ---------------------------------------------------------------- 1
def test_the_arc_keeps_its_length_on_target_and_grows_back_from_a_stall():
    from vfife.implicit import resize_arc
    # 6 iterations against a target of 5 kept shrinking the arc by sqrt(5/6)
    assert resize_arc(1.0, 6, 0, 5, 0, 1.0) == (1.0, 1)
    dl, good = resize_arc(1.0, 8, 0, 5, 0, 1.0)
    assert dl == pytest.approx(np.sqrt(5 / 8)) and good == 1
    # a quarter of the first arc, three increments in a row without a cutback:
    # the iteration count is not answering to the arc, so it is doubled
    dl, good = resize_arc(0.1, 7, 0, 5, 2, 1.0)
    assert dl == pytest.approx(0.2) and good == 0
    # a cutback resets the run and never grows the arc
    assert resize_arc(0.1, 2, 1, 5, 2, 1.0) == (0.1, 0)


def test_running_out_of_increments_does_not_ask_for_the_stop_it_has(tmp_path):
    case = api.load_case(ROOT / "cases" / "case_16b_lee_frame_arc_length.yaml")
    c = case.with_changes({"solver.static.arc_length.max_increments": 3,
                           "solver.static.arc_length.stop": {"region": "load", "dof": "z", "value": -2.0}})
    out = api.run_case(c, output=str(tmp_path / "lee.h5"))
    assert out.status == "MAX_STEPS"
    assert "set a stop" not in out.message and "of -2 at load" in out.message


# ---------------------------------------------------------------- 2
def test_a_static_run_writes_every_increment_unless_told_otherwise(tmp_path):
    """`output.every` defaults to 200 and counted increments: a 90-position
    influence sweep wrote only its last frame (the truck off the span)."""
    _, h5, _ = _run(tmp_path, _strut(steps=[{"name": "s", "load_steps": 5}]), "dense")
    with h5py.File(h5) as f:
        assert len(f["steps"]) == 5
    _, h5, _ = _run(tmp_path, _strut(steps=[{"name": "s", "load_steps": 5}], every=200), "thin")
    with h5py.File(h5) as f:
        assert len(f["steps"]) == 1 and len(f["monitors/time"]) == 5


# ---------------------------------------------------------------- 4
def test_member_force_monitors_on_a_frame(tmp_path):
    """`kind: element` was refused on line models (MONITOR_UNSUPPORTED): a
    hanger force history needed frames every 50 steps, 78 MB for 49 members."""
    mons = {"N": {"kind": "element", "region": "beam", "field": "axial_force", "reduce": "absmax"},
            "M": {"kind": "element", "region": "beam", "field": "bending_y_max_abs"}}
    out, h5, _ = _run(tmp_path, _strut(steps=[{"name": "s", "load_steps": 4}], monitors=mons), "mon")
    assert out.converged
    np.testing.assert_allclose(np.asarray(out.monitors["N"]), -1.0e5 * np.arange(1, 5) / 4, rtol=1e-3)
    Mabs, = _last(h5, "bending_y_max_abs")
    assert out.monitors["M"][-1] == pytest.approx(float(Mabs.max()), rel=1e-6)


def test_a_monitor_reading_an_unknown_field_or_a_needless_reduce_is_refused(tmp_path):
    d = _strut(monitors={"N": {"kind": "element", "region": "beam", "field": "axial_forse"}})
    with pytest.raises(UserError) as e:
        _run(tmp_path, d, "badmon")
    assert e.value.code == "UNKNOWN_MONITOR_FIELD"
    d = _strut(monitors={"w": {"kind": "displacement", "region": "mid", "dof": "z", "reduce": "max"}})
    p = tmp_path / "red.yaml"
    d["name"] = "red"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    issues = api.prepare_case(str(p)).preflight["issues"]
    msg = next(i["message"] for i in issues if i["code"] == "MONITOR_REDUCE_UNUSED")
    assert "mean displacement of its regions' nodes" in msg and "element monitors" in msg


# ---------------------------------------------------------------- 5
def test_a_removed_member_reports_no_force(tmp_path):
    """A cut hanger split off as a free rod that rang at +-650 kN, reported as
    `axial_force` +646.7 kN (compression shown positive) in a failed member."""
    d = _hanging(tmp_path, 10.0, removal=0.05, total=0.12)
    d["output"]["fields"] = ["displacement", "axial_force", "member_failed"]
    d["output"]["monitors"]["N"] = {"kind": "element", "region": "cable", "field": "axial_force"}
    out, h5, _ = _run(tmp_path, d, "cut")
    N, failed = _last(h5, "axial_force", "member_failed")
    assert failed[1] == 1.0 and N[1] == 0.0
    mon = np.asarray(out.monitors["N"])
    assert mon[0] > 0 and mon[-1] == 0.0                      # the cable in tension, then nothing


# ---------------------------------------------------------------- 6
def test_view_joins_runs_of_one_model_with_different_spare_particles(tmp_path):
    """A run that can split reserves spare particles: the dead state and the
    cut of one bridge were refused as "a different mesh"."""
    from vfife.post import view
    plain = _hanging(tmp_path, 10.0, total=0.02)
    plain["assign"][1].pop("failure")
    _, h_plain, _ = _run(tmp_path, plain, "plain")
    _, h_cut, _ = _run(tmp_path, _hanging(tmp_path, 10.0, removal=0.01, total=0.02), "cut2")
    with h5py.File(h_plain) as a, h5py.File(h_cut) as b:
        assert len(a["mesh/nodes"]) < len(b["mesh/nodes"])
    data = view.load([str(h_plain), str(h_cut)])
    assert len(data["meta"]["acts"]) == 2


# ---------------------------------------------------------------- 7
def test_a_continuation_says_which_table_rows_were_already_written(tmp_path):
    """A continuation's member_failures holds the whole history (the
    checkpoint carries it) while its frames hold the tail: joining the two
    files counted the early breaks twice."""
    d = _hanging(tmp_path, 10.0, removal=0.03, total=0.12)
    d["output"]["checkpoint_every"] = 200
    seen = {"n": 0}

    def stop(step, n, *a):
        seen["n"] += 1
        return not (step > 0.6 * n)
    with pytest.raises(api.AnalysisInterrupted):
        _run(tmp_path, d, "part", on_progress=stop, progress_every=50)
    rest = tmp_path / "rest.h5"
    api.run_case(str(tmp_path / "part.yaml"), output=str(rest), restart=str(tmp_path / "part.h5"))
    with h5py.File(tmp_path / "part.h5") as a, h5py.File(rest) as b:
        pa, pb = a["diagnostics/member_failures"], b["diagnostics/member_failures"]
        assert pa.shape[0] == 1 and pa.attrs["rows_before_restart"] == 0
        assert pb.shape[0] == 1 and pb.attrs["rows_before_restart"] == 1
        assert b.attrs["restarted_at_time"] > 0.03


# ---------------------------------------------------------------- 9
def test_probe_skips_the_force_free_and_stability_checks_of_a_self_stressed_model():
    """`probe` said FAIL "the reference state is force-free" on every
    prestressed girder, and FAIL "the state is stable" at an arc of it."""
    from vfife.cli import _with_element, build, make_solver
    from vfife.io import schema
    from vfife.testing import invariants
    from vfife.testing.state import probe as make_probe
    text = (ROOT / "cases" / "case_29g_prestress_loss.yaml").read_text(encoding="utf-8")
    why = invariants.self_stress(text)
    assert "tendons" in why and "creep" in why
    assert invariants.self_stress((ROOT / "cases" / "case_01_beam_cantilever.yaml").read_text()) == ""
    path = str(ROOT / "cases" / "case_01_beam_cantilever.yaml")
    case, _ = schema.load(path)
    mesh, material, element = build(case, path)
    p = make_probe(make_solver(_with_element(case, element, path), mesh, material))
    found = {f.name: f for f in invariants.report(p, np.zeros(p.n), self_stressed=why)}
    for name in ("the reference state is force-free", "the state is stable"):
        assert found[name].skipped and "self-stressed" in found[name].detail


# ---------------------------------------------------------------- 11, 12
def test_the_debug_hint_names_the_program_option_and_the_unit_hint_needs_a_number(tmp_path, capsys):
    """The hint said "run again with --debug" (`vfife view ... --debug` is
    refused: it is the program's option) and added a unit-system hint to
    errors that are not about a number's size."""
    from vfife import cli
    _, h1, _ = _run(tmp_path, _strut(), "s1")
    _, h2, _ = _run(tmp_path, _hanging(tmp_path, 10.0, total=0.01), "h1")
    rc = cli.main(["view", str(h1), str(h2), "-o", str(tmp_path / "x.html")])
    err = capsys.readouterr().err
    assert rc == 1 and "a different mesh" in err
    assert "vfife --debug view" in err and "unit system" not in err


def test_run_reports_an_unknown_output_field_as_a_preflight_failure(tmp_path, capsys):
    """`run` printed "preflight PASS" and then refused the case for
    UNKNOWN_OUTPUT_FIELD, which `check` reports as preflight FAIL."""
    from vfife import cli
    d = _strut(fields=["displacement", "axial_forse"])
    p = tmp_path / "bad.yaml"
    d["name"] = "bad"
    d["output"]["path"] = str(tmp_path / "bad.h5")
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    rc = cli.main(["run", str(p)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "preflight     FAIL" in out and "UNKNOWN_OUTPUT_FIELD" in out and "preflight     PASS" not in out


# ---------------------------------------------------------------- 13
def test_a_limit_message_says_how_far_the_loads_got():
    """"load factor 0.725" was the step's progress; with `push: [[0, 0], [1, 12]]`
    the loads stood at 8.7 times their written value."""
    from vfife.core import amplitude as amp
    from vfife.implicit import amplitude_note

    class S:
        amplitudes = amp.build({"push": [[0.0, 0.0], [1.0, 12.0]]})
    groups = {"push": np.ones(3), "hold": np.ones(3)}
    assert amplitude_note(S(), groups, 0.725) == " (step loads at push x8.7 of their written value)"
    assert "push x6" in amplitude_note(S(), groups, 0.5, pattern=True)
    assert amplitude_note(S(), {"hold": np.ones(3)}, 0.5) == ""


# ---------------------------------------------------------------- 14
def test_a_static_removal_is_refused_for_its_own_mode(tmp_path):
    d = _strut()
    d["removals"] = [{"region": "mid", "time": 1.0}]
    p = tmp_path / "rem.yaml"
    d["name"] = "rem"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    with pytest.raises(Exception) as e:
        api.prepare_case(str(p))
    # since 1.7 a static step takes member failure, but a removal at a TIME
    # has no clock there: refused for the mode it is, with the way out
    assert "mode: static" in str(e.value) and "deactivate" in str(e.value)


def test_a_restart_refusal_names_the_key_that_changed():
    from vfife.io.fingerprint import _named
    a = json.dumps({"material": {"type": "concrete02", "creep": {"model": "mc90", "rh": 70.0}}})
    b = json.dumps({"material": {"type": "concrete02", "creep": {"model": "mc90", "rh": 50.0}}})
    assert _named(a, b) == ["material.creep.rh (70.0 there, 50.0 here)"]
