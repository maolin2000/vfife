"""Each fibre segment on its own concrete clock (1.8).

A fibre model with creep or shrinkage could not change its model in an
analysis step (1.2): the creep scale was per material, so a segment born
late would have shrunk and crept from the model's first day. It is per
member now -- `staging.age_shift` and `staging.age_floor` (`_age_shift`, `_age_floor` until v1.18), the frames' rule since 1.2 --
on the reference path (the law's strain scale set member by member) and in
the fused kernel (its per-member parameters).

The fibre twins of tests/test_v12_steps.py's frame checks, on concrete02
fibres in compression at 6% of their strength (the elastic stretch is the
parabola's, the creep 1 + phi times it):

* two independent bars under a held force, the second cast when the
  model's clock reads 100 days at its own age of 7 days: each stretches by
  1 + phi on ITS clock, phi(1000, 28) and phi(907, 7);
* a free bar cast late shortens by eps_sh(907) - eps_sh(7) -- what it shrank
  in the formwork is not a load;
* a held stretch on a late bar relaxes station for station like a bar
  loaded at its own age in a model of its own;
* the reference path and the kernel agree.
"""
import numpy as np
import pytest

from vfife.api import run_case
from vfife.core.creep import creep_coefficient, shrinkage_strain

CONC = {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.005,
        "ft": 2.0e6, "Ets": 1.0e9, "rat": 0.1, "rho": 2400}
A = 0.3 * 0.4
LB = 10.0


def _run(d):
    out = run_case(d)
    assert out.converged, out.message
    return out


def _bars(tmp_path, name, steps, age0=28.0, laws=None, backend="numpy", monitors=None):
    mat = dict(CONC, **(laws or {"creep": {"model": "aci209"}}))
    return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "lines",
                     "points": {"A": [0, 0, 0], "B": [LB, 0, 0], "C": [0, 5, 0], "D": [LB, 5, 0]},
                     "lines": {"b1": {"through": ["A", "B"], "size": 2.5},
                               "b2": {"through": ["C", "D"], "size": 2.5}}},
            "materials": {"c": mat},
            "sections": {"g": {"type": "fiber", "material": "c", "shape": "rect", "width": 0.3, "height": 0.4,
                               "ny": 2, "nz": 6}},
            "assign": [{"region": "all", "section": "g", "orientation": [0, 0, 1]}],
            "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "C", "dofs": [1] * 6},
                            {"region": "all", "dofs": [0, 1, 1, 1, 1, 1]}],
            "loads": [], "steps": steps,
            "solver": {"mode": "relaxation", "backend": backend, "tolerance": 1e-9, "kinetic_tolerance": 1e-14,
                       "age": age0},
            "output": {"path": str(tmp_path / f"{name}.h5"), "fields": ["displacement"],
                       "monitors": monitors or {"uB": {"kind": "displacement", "region": "B", "dof": "x"},
                                                "uD": {"kind": "displacement", "region": "D", "dof": "x"}}}}


N = -2.0e5            # compression, 1.7 MPa: 6% of the strength


def _two_bars(tmp_path, name, backend="numpy"):
    return _bars(tmp_path, name, [
        {"name": "load1", "load_steps": 1, "max_steps_per_load": 40000, "activate": ["b1"],
         "loads": [{"region": "B", "vector": [N, 0, 0], "keep": True}]},
        {"name": "creep1", "age": 28.0, "to_age": 100.0, "load_steps": 12, "max_steps_per_load": 40000},
        {"name": "cast2", "activate": ["b2"], "activate_age": 7.0, "load_steps": 1,
         "max_steps_per_load": 40000, "loads": [{"region": "D", "vector": [N, 0, 0], "keep": True}]},
        {"name": "creep2", "to_age": 1000.0, "load_steps": 24, "max_steps_per_load": 40000}],
        backend=backend)


def test_a_fibre_segment_born_late_creeps_on_its_own_clock(tmp_path):
    out = _run(_two_bars(tmp_path, "age"))
    spec = {"model": "aci209"}
    uB, uD = np.asarray(out.monitors["uB"]), np.asarray(out.monitors["uD"])
    # the elastic stretch of each bar, read off the run (concrete02's secant)
    e1, e2 = uB[0], uD[13]
    # concrete02's parabola: fc (2 eta - eta^2) A = |N|
    eta = 1.0 - np.sqrt(1.0 - abs(N) / (3.0e7 * A))
    assert e1 == pytest.approx(-eta * 0.002 * LB, rel=1e-6)
    assert e2 == pytest.approx(e1, rel=1e-6)                       # born free, then loaded alike
    assert uB[-1] / e1 == pytest.approx(1 + float(creep_coefficient(spec, 1000.0, 28.0)), rel=5e-3)
    assert uD[-1] / e2 == pytest.approx(1 + float(creep_coefficient(spec, 907.0, 7.0)), rel=5e-3)
    # the late bar crept more: it was loaded younger
    assert uD[-1] / e2 > uB[-1] / e1 * 1.05


def test_a_fibre_segment_born_late_shrinks_from_its_birth(tmp_path):
    spec = {"model": "aci209"}
    d = _bars(tmp_path, "sh", [
        {"name": "s0", "age": 28.0, "to_age": 100.0, "load_steps": 4, "max_steps_per_load": 40000},
        {"name": "cast2", "activate": ["b2"], "activate_age": 7.0, "load_steps": 1, "max_steps_per_load": 40000},
        {"name": "s1", "to_age": 1000.0, "load_steps": 6, "max_steps_per_load": 40000}],
        laws={"shrinkage": spec})
    out = _run(d)
    uB, uD = np.asarray(out.monitors["uB"]), np.asarray(out.monitors["uD"])
    assert uD[4] == pytest.approx(0.0, abs=1e-12)                  # born free, where it was cast
    assert uD[-1] == pytest.approx(LB * (shrinkage_strain(spec, 907.0) - shrinkage_strain(spec, 7.0)), rel=1e-4)
    assert uB[-1] == pytest.approx(LB * shrinkage_strain(spec, 1000.0), rel=1e-4)


def test_a_late_fibre_segment_relaxes_like_a_bar_loaded_at_its_age(tmp_path):
    hold = {"region": "D", "dofs": [1, 0, 0, 0, 0, 0], "value": -1.0e-4, "keep": True}
    mon = {"RD": {"kind": "reaction", "region": "D", "dof": "x"}}
    born = _run(_bars(tmp_path, "born", [
        {"name": "s0", "age": 28.0, "to_age": 100.0, "load_steps": 4, "max_steps_per_load": 40000},
        {"name": "cast", "activate": ["b2"], "activate_age": 7.0, "load_steps": 1,
         "max_steps_per_load": 40000, "constraints": [hold]},
        {"name": "relax", "to_age": 1000.0, "load_steps": 12, "time_spacing": "linear",
         "max_steps_per_load": 40000}], monitors=mon))
    ref = _run(_bars(tmp_path, "ref", [
        {"name": "load", "load_steps": 1, "max_steps_per_load": 40000, "constraints": [hold],
         "deactivate": ["b1"]},
        {"name": "relax", "age": 7.0, "to_age": 907.0, "load_steps": 12, "time_spacing": "linear",
         "max_steps_per_load": 40000}], age0=7.0, monitors=mon))
    b, r = np.asarray(born.monitors["RD"][-12:]), np.asarray(ref.monitors["RD"][-12:])
    assert b == pytest.approx(r, rel=1e-6)
    assert abs(b[-1]) < 0.7 * abs(b[0])                           # it did relax


def test_the_kernel_carries_each_segment_s_clock_as_the_reference_does(tmp_path):
    pytest.importorskip("numba")
    a = _run(_two_bars(tmp_path, "np", "numpy"))
    b = _run(_two_bars(tmp_path, "nb", "numba"))
    for k in ("uB", "uD"):
        assert np.asarray(b.monitors[k]) == pytest.approx(np.asarray(a.monitors[k]), rel=1e-8)


def test_a_fibre_model_whose_tendons_relax_still_may_not_change(tmp_path):
    """A tendon's relaxation is dated from the model's stressing age: a late
    one is refused by name, as before."""
    d = _bars(tmp_path, "rx", [{"name": "s", "activate": ["b2"], "load_steps": 1, "max_steps_per_load": 100}])
    d["materials"]["p"] = {"type": "steel02", "E": 1.95e11, "Fy": 1.6e9, "b": 0.01, "rho": 7850,
                           "relaxation": {"model": "power", "rho_1000": 0.031, "k": 0.21}}
    d["sections"]["strand"] = {"type": "bar", "material": "p", "shape": "general", "area": 1.4e-4}
    d["sections"]["g"]["tendons"] = [{"section": "strand", "profile": [[0.0, 0.0, -0.1], [LB, 0.0, -0.1]],
                                      "prestress": 1.2e9}]
    with pytest.raises(Exception, match="tendons relax"):
        run_case(d)


def test_a_run_stopped_while_the_late_segment_creeps_resumes_where_it_was(tmp_path):
    from vfife.core.diagnostics import AnalysisInterrupted
    full = _run(_two_bars(tmp_path, "full"))
    d = _two_bars(tmp_path, "stop")
    calls = {"n": 0}

    def stop(*_a):
        calls["n"] += 1
        return calls["n"] < 30            # inside creep2: 1 + 12 + 1 levels before it

    with pytest.raises(AnalysisInterrupted):
        run_case(d, on_progress=stop, progress_every=10 ** 6)
    d2 = _two_bars(tmp_path, "rest")
    rest = run_case(d2, restart=str(tmp_path / "stop.h5"), progress_every=10 ** 6)
    assert rest.converged, rest.message
    for k in ("uB", "uD"):
        assert np.asarray(rest.monitors[k])[-1] == np.asarray(full.monitors[k])[-1]


def test_a_segment_born_in_a_step_that_moves_the_clock_is_a_bar_built_at_its_age(tmp_path):
    """Born, loaded and creeping and shrinking in one step that runs the
    clock from 100 to 1000 days: the bar is the one of a model of its own
    whose clock starts at its age 7 (less the shrinkage it had by then)
    -- the first station after the birth dated from the birth, the windows
    on the bar's clock (found in review)."""
    sp = {"model": "aci209"}
    laws = {"creep": sp, "shrinkage": sp}
    mon = {"uD": {"kind": "displacement", "region": "D", "dof": "x"}}
    load = {"region": "D", "vector": [N, 0, 0], "keep": True}
    born = _run(_bars(tmp_path, "born", [
        {"name": "s0", "age": 28.0, "to_age": 100.0, "load_steps": 3, "max_steps_per_load": 40000},
        {"name": "cast", "activate": ["b2"], "activate_age": 7.0, "to_age": 1000.0, "load_steps": 12,
         "time_spacing": "linear", "max_steps_per_load": 40000, "loads": [load]}], laws=laws, monitors=mon))
    ref = _run(_bars(tmp_path, "ref", [
        {"name": "z", "deactivate": ["b1"], "load_steps": 1, "max_steps_per_load": 40000},
        {"name": "cast", "to_age": 907.0, "load_steps": 12, "time_spacing": "linear",
         "max_steps_per_load": 40000, "loads": [load]}], age0=7.0, laws=laws, monitors=mon))
    b = np.asarray(born.monitors["uD"])[-12:]
    r = np.asarray(ref.monitors["uD"])[-12:] - LB * shrinkage_strain(sp, 7.0)
    assert b == pytest.approx(r, rel=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_segment_taken_out_and_cast_again_is_new_concrete_on_a_new_clock(tmp_path, backend):
    """Loaded, crept, taken out, cast again at age 7: from the second birth
    the bar is the one born only then, to the bit."""
    if backend == "numba":
        pytest.importorskip("numba")
    laws = {"creep": {"model": "aci209"}, "shrinkage": {"model": "aci209"}}
    ld = {"region": "D", "vector": [N, 0, 0]}
    tail = [{"name": "cast2", "activate": ["b2"], "activate_age": 7.0, "load_steps": 1,
             "max_steps_per_load": 40000, "loads": [dict(ld, keep=True)]},
            {"name": "creep2", "to_age": 1000.0, "load_steps": 8, "max_steps_per_load": 40000}]
    again = _run(_bars(tmp_path, "again", [
        {"name": "l1", "load_steps": 1, "max_steps_per_load": 40000, "loads": [ld]},
        {"name": "c1", "age": 28.0, "to_age": 100.0, "load_steps": 4, "max_steps_per_load": 40000, "loads": [ld]},
        {"name": "off", "deactivate": ["b2"], "to_age": 200.0, "load_steps": 2, "max_steps_per_load": 40000}]
        + tail, laws=laws, backend=backend))
    once = _run(_bars(tmp_path, "once", [
        {"name": "l1", "load_steps": 1, "max_steps_per_load": 40000, "deactivate": ["b2"]},
        {"name": "c1", "age": 28.0, "to_age": 100.0, "load_steps": 4, "max_steps_per_load": 40000},
        {"name": "off", "to_age": 200.0, "load_steps": 2, "max_steps_per_load": 40000}] + tail,
        laws=laws, backend=backend))
    a, b = np.asarray(again.monitors["uD"]), np.asarray(once.monitors["uD"])
    assert (a[-9:] - a[-10]) == pytest.approx(b[-9:] - b[-10], rel=1e-7)


def test_a_member_not_built_does_not_size_the_residual(tmp_path):
    """Its shrinkage is no restraint: with one of two equal bars taken out
    the residual's scale is half the one with both (found in review,
    v1.8)."""
    import yaml
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver
    from vfife.io.writer import read_checkpoint
    sp = {"model": "aci209"}
    scale = {}
    for name, off in (("both", {}), ("one", {"deactivate": ["b2"]})):
        d = _bars(tmp_path, name, [dict({"name": "s0", "age": 28.0, "to_age": 100.0, "load_steps": 2,
                                         "max_steps_per_load": 40000}, **off)], laws={"shrinkage": sp})
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
        _run(d)
        prep = prepare_case(str(p))
        s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)
        s.restore(read_checkpoint(str(tmp_path / f"{name}.h5")))
        if off:
            assert s.member_built.sum() == len(s.member_built) // 2      # b2's members out
        s._refresh_imposed(1.0)
        scale[name] = s._thermal_scale
    assert scale["one"] == pytest.approx(0.5 * scale["both"], rel=1e-12)
