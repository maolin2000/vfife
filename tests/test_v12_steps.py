"""1.2, item 9: analysis steps -- each step its own procedure, members born
and removed, loads and constraints that outlive their step.

Every check is a closed form of a beam built in stages, or the statement
that a change of procedure moves nothing:

* two equal spans cast one after the other: the first carries its own
  weight simply supported (R_A = wL/2), the second's weight goes onto the
  two-span continuous beam (R_A = -wL/16 of it): R_A = 7/16 wL, R_B = 9/8 wL,
  against 3/8 and 5/4 cast at once;
* a member removed takes its forces off: the first span alone, simply
  supported, R_A = R_B = wL/2;
* a support released: the two spans become one, R_A = R_C = wL;
* a relaxed dead-load state handed to a dynamic step with nothing new on it
  does not move.
"""
import h5py
import numpy as np
import pytest

from vfife.api import run_case

L = 20.0
W = 2500.0 * 1.5 * 9.80665          # rho A g, N/m


def _case(tmp_path, name, steps, constraints=None, mode="relaxation", loads=None, extra=None):
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [L, 0, 0], "C": [2 * L, 0, 0]},
                  "lines": {"s1": {"through": ["A", "B"], "size": 2.0},
                            "s2": {"through": ["B", "C"], "size": 2.0}}},
         "materials": {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2500.0}},
         "sections": {"g": {"type": "beam", "material": "c", "shape": "rect", "width": 1.0,
                            "height": 1.5}},
         "assign": [{"region": "all", "section": "g", "orientation": [0, 1, 0]}],
         "constraints": constraints if constraints is not None else [
             {"region": "A", "dofs": [1, 1, 1, 1, 0, 0]}, {"region": "B", "dofs": [0, 1, 1, 0, 0, 0]},
             {"region": "C", "dofs": [0, 1, 1, 0, 0, 0]}],
         "loads": loads if loads is not None else [{"region": "all", "gravity": [0, 0, -9.80665]}],
         "steps": steps,
         "solver": {"mode": mode, "backend": "numpy", "tolerance": 1e-7},
         "output": {"path": str(tmp_path / f"{name}.h5"), "monitors": {
             "RA": {"kind": "reaction", "region": "A", "dof": "z"},
             "RB": {"kind": "reaction", "region": "B", "dof": "z"},
             "RC": {"kind": "reaction", "region": "C", "dof": "z"},
             "w": {"kind": "displacement", "region": "all", "dof": "z"}}}}
    d.update(extra or {})
    return d


def _st(name, **kw):
    st = {"name": name, "load_steps": 4, "max_steps_per_load": 40000}
    st.update(kw)
    return st


def _run(d):
    out = run_case(d)
    assert out.converged, out.message
    return out


def _r(out, key):
    return out.monitors[key][-1] / (W * L)


def test_two_spans_cast_one_after_the_other(tmp_path):
    once = _run(_case(tmp_path, "once", [_st("all")]))
    staged = _run(_case(tmp_path, "staged", [_st("span1", activate=["s1"]),
                                              _st("span2", activate=["s2"])]))
    # the closed forms, to the lumped self-weight of a 2 m mesh
    assert _r(once, "RA") == pytest.approx(3 / 8, rel=4e-3)
    assert _r(staged, "RA") == pytest.approx(7 / 16, rel=2e-3)
    assert _r(staged, "RB") == pytest.approx(9 / 8, rel=2e-3)
    # the first span carried itself simply supported: R_A = wL/2 when it was done
    assert staged.monitors["RA"][3] / (W * L) == pytest.approx(0.5, rel=1e-6)
    # and the second span's weight came on over its step, not at once
    rc = np.asarray(staged.monitors["RC"][4:]) / (W * L)
    assert np.all(np.diff(rc) > 0) and rc[0] == pytest.approx(rc[-1] / 4, rel=1e-3)
    # the members not yet built are written as such
    with h5py.File(staged.result_path, "r") as f:
        frames = sorted(f["steps"], key=int)
        first = np.asarray(f["steps"][frames[0]]["member_built"])
        last = np.asarray(f["steps"][frames[-1]]["member_built"])
        assert first.sum() == 10 and last.sum() == 20
        assert f["steps"][frames[0]].attrs["procedure"] == "relaxation"


def test_static_steps_give_the_relaxation_s_answer(tmp_path):
    steps = [{"name": "span1", "activate": ["s1"], "load_steps": 4},
             {"name": "span2", "activate": ["s2"], "load_steps": 4}]
    relax = _run(_case(tmp_path, "relax", [_st("span1", activate=["s1"]), _st("span2", activate=["s2"])]))
    static = _run(_case(tmp_path, "static", [dict(s) for s in steps], mode="static"))
    mixed = _run(_case(tmp_path, "mixed", [dict(s, procedure="static") for s in steps]))
    for out in (static, mixed):
        for key in ("RA", "RB", "RC"):
            assert _r(out, key) == pytest.approx(_r(relax, key), rel=1e-6)


def test_a_dead_load_state_handed_to_a_dynamic_step_does_not_move(tmp_path):
    out = _run(_case(tmp_path, "still", [_st("dead"),
                                         {"name": "quake", "procedure": "dynamic", "total_time": 0.5}]))
    w = np.asarray(out.monitors["w"])
    dead = w[3]
    assert np.ptp(w[4:]) < 1e-9 * abs(dead)
    with h5py.File(out.result_path, "r") as f:
        assert f.attrs["time_axis"] == "time"
        frames = sorted(f["steps"], key=int)
        t = [float(f["steps"][k].attrs["time"]) for k in frames]
        procs = [f["steps"][k].attrs["procedure"] for k in frames]
    # the relaxation's levels sit at t = 0, the clock the dynamic step moves
    assert t[0] == 0.0 and t[-1] == pytest.approx(0.5) and np.all(np.diff(t) >= 0)
    assert procs[0] == "relaxation" and procs[-1] == "dynamic"


def test_a_removed_member_takes_its_forces_off(tmp_path):
    out = _run(_case(tmp_path, "cut", [_st("both"), _st("cut", deactivate=["s2"])]))
    assert _r(out, "RA") == pytest.approx(0.5, rel=1e-6)
    assert _r(out, "RB") == pytest.approx(0.5, rel=1e-6)
    assert abs(_r(out, "RC")) < 1e-9
    # gradually: the reaction at C goes to zero over the step's levels
    rc = np.asarray(out.monitors["RC"][4:]) / (W * L)
    assert np.all(np.diff(rc) < 0)


def test_a_released_support_hands_its_reaction_back(tmp_path):
    cons = [{"region": "A", "dofs": [1, 1, 1, 1, 0, 0]}, {"region": "C", "dofs": [0, 1, 1, 0, 0, 0]}]
    out = _run(_case(tmp_path, "rel", [
        _st("temp", constraints=[{"region": "B", "dofs": [0, 1, 1, 0, 0, 0], "keep": True}]),
        _st("free", release=["B"], max_steps_per_load=80000)], constraints=cons))
    assert out.monitors["RB"][3] / (W * L) == pytest.approx(1.25, rel=4e-3)   # held: two spans
    assert _r(out, "RA") == pytest.approx(1.0, rel=1e-6)
    assert abs(_r(out, "RB")) < 1e-9
    assert _r(out, "RC") == pytest.approx(1.0, rel=1e-6)


def test_a_kept_load_stays_and_a_step_s_own_does_not(tmp_path):
    out = _run(_case(tmp_path, "keep", [
        _st("p", load_steps=2, loads=[{"region": "B", "vector": [0, 0, -1e6], "keep": True},
                                      {"region": "C", "vector": [0, 0, -2e5]}]),
        _st("q", load_steps=2)], loads=[]))
    assert out.monitors["RB"] == pytest.approx([5e5, 1e6, 1e6, 1e6], rel=1e-6)
    assert out.monitors["RC"][1] == pytest.approx(2e5, rel=1e-6) and abs(out.monitors["RC"][-1]) < 1e-3


def _cantilever(tmp_path, name, birth):
    d = _case(tmp_path, name, [_st("seg1", activate=["s1"]),
                               _st("seg2", activate=["s2"], birth=birth)],
              constraints=[{"region": "A", "dofs": [1] * 6}])
    d["output"]["monitors"] = {"wB": {"kind": "displacement", "region": "B", "dof": "z"},
                               "wC": {"kind": "displacement", "region": "C", "dof": "z"}}
    return _run(d)


def test_a_cantilever_cast_on_its_tangent_and_one_cast_to_its_drawing(tmp_path):
    once = _case(tmp_path, "c_once", [_st("all")], constraints=[{"region": "A", "dofs": [1] * 6}])
    once["output"]["monitors"] = {"wC": {"kind": "displacement", "region": "C", "dof": "z"}}
    once = _run(once)
    tangent = _cantilever(tmp_path, "c_t", "tangent")
    design = _cantilever(tmp_path, "c_d", "design")
    # on the tangent, a linear cantilever ends where it would have cast at once
    assert tangent.monitors["wC"][-1] == pytest.approx(once.monitors["wC"][-1], rel=2e-4)
    # to the drawing, the tip starts at its design point: it is short of the
    # one-shot answer by what the first segment's deflection and rotation at B
    # would have carried it, w_B + theta_B L
    EI = 3.0e10 * 1.0 * 1.5 ** 3 / 12
    wB, thB = W * L ** 4 / (8 * EI), W * L ** 3 / (6 * EI)
    assert design.monitors["wC"][-1] - once.monitors["wC"][-1] == pytest.approx(wB + thB * L, rel=5e-3)   # lumped weight
    # the second segment's tip, before its own weight, was on the tangent
    assert tangent.monitors["wC"][4] < design.monitors["wC"][4]


def test_a_dynamic_step_with_members_still_to_build(tmp_path):
    """Dormant nodes are held, and the stable step is the reduced masses'."""
    out = _run(_case(tmp_path, "dorm", [_st("span1", activate=["s1"]),
                                        {"name": "shake", "procedure": "dynamic", "total_time": 0.2}]))
    w = np.asarray(out.monitors["w"])
    assert np.ptp(w[4:]) < 1e-9 * abs(w[3])


@pytest.mark.parametrize("steps, words", [
    ([_st("a", activate=["s1"]), _st("b", activate=["s1"])], "already built"),
    ([_st("a", deactivate=["s1"]), _st("b", deactivate=["s1"])], "not built"),
    ([_st("a", activate=["A"])], "holds no complete member"),
    ([_st("a", total_time=1.0)], "total_time is a dynamic step's duration"),
    ([_st("a", activate=["s1"], deactivate=["s1"])], "both activated and deactivated"),
    ([_st("a", activate_age=7.0)], "activates none"),
])
def test_what_steps_refuse(tmp_path, steps, words):
    with pytest.raises(ValueError, match=words):
        run_case(_case(tmp_path, "bad", steps))


def test_keep_is_a_step_s_and_a_dynamic_step_is_not_kinetic(tmp_path):
    d = _case(tmp_path, "k", [_st("a")], loads=[{"region": "B", "vector": [0, 0, -1.0], "keep": True}])
    with pytest.raises(ValueError, match="`keep` on a model-level"):
        run_case(d)
    d = _case(tmp_path, "kd", [_st("a"), {"name": "b", "procedure": "dynamic", "total_time": 0.1}],
              extra={"solver": {"mode": "relaxation", "backend": "numpy", "tolerance": 1e-7,
                                "damping": {"type": "kinetic"}}})
    with pytest.raises(ValueError, match="kinetic"):
        run_case(d)


def test_a_model_change_on_a_solid_is_refused_by_name():
    from vfife.io.schema import Case
    with pytest.raises(ValueError, match="line models"):
        Case.model_validate({
            "mesh": {"type": "box", "dimensions": [1, 1, 1], "divisions": [1, 1, 1]},
            "material": {"type": "linear_elastic", "E": 1.0, "nu": 0.2, "rho": 1.0},
            "solver": {"mode": "relaxation", "n_steps": 10},
            "steps": [{"name": "a", "procedure": "dynamic", "total_time": 1.0}]})


def test_a_staged_run_stopped_in_a_model_changing_step_resumes(tmp_path):
    """N levels in one run, and stopped in the second span's step and
    restarted, end in the same state: what was built, the birth offsets,
    the masses and the unloading forces are in the checkpoint."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_restart_resumes import _same, _split
    d = _case(tmp_path, "rs", [_st("span1", activate=["s1"]),
                               _st("span2", activate=["s2"], deactivate=[]),
                               {"name": "shake", "procedure": "dynamic", "total_time": 0.05}])
    d.pop("name"); d["output"].pop("path")
    full, half, rest = _split(tmp_path, d, lambda s, stage: s.step_index == 1 and stage >= 2,
                              progress_every=50)
    _same(full, rest)
    assert rest.member_built.all() and rest.times[-1] == pytest.approx(full.times[-1])


def test_a_run_stopped_in_its_dynamic_step_resumes_on_its_clock(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_restart_resumes import _same, _split
    d = _case(tmp_path, "rd", [_st("span1"), {"name": "shake", "procedure": "dynamic", "total_time": 0.05,
                                              "loads": [{"region": "B", "vector": [0, 0, -1e5]}]}])
    d.pop("name"); d["output"].pop("path")
    full, half, rest = _split(tmp_path, d, lambda s, local: s.step_index == 1 and local >= 40,
                              progress_every=20)
    # the relaxation's last convergence measures are not the dynamic step's
    _same(full, rest, skip=("loop_measures",))
    assert rest.times[-1] == pytest.approx(0.05) and rest.mode == "dynamic"


# --------------------------------------------------------------------------- #
#  from the review (1.2)
# --------------------------------------------------------------------------- #

def _three(tmp_path, name, steps, loads, constraints=None, extra=None):
    d = _case(tmp_path, name, steps, loads=loads, extra=extra, constraints=constraints if constraints is not None else [
        {"region": "A", "dofs": [1, 1, 1, 1, 0, 0]}, {"region": "B", "dofs": [0, 1, 1, 0, 0, 0]},
        {"region": "C", "dofs": [0, 1, 1, 0, 0, 0]}, {"region": "D", "dofs": [0, 1, 1, 0, 0, 0]}])
    d["mesh"]["points"]["D"] = [3 * L, 0, 0]
    d["mesh"]["points"]["M1"] = [L / 2, 0, 0]
    d["mesh"]["lines"] = {"s1": {"through": ["A", "M1", "B"], "size": 2.0},
                          "s2": {"through": ["B", "C"], "size": 2.0},
                          "s3": {"through": ["C", "D"], "size": 2.0}}
    d["output"]["monitors"]["RD"] = {"kind": "reaction", "region": "D", "dof": "z"}
    return d


def test_a_member_born_twice_is_born_without_stress_both_times(tmp_path):
    P = [{"region": "M1", "vector": [0, 0, -1e5]}]
    out = _run(_three(tmp_path, "twice", [_st("a"), _st("b", deactivate=["s2"]), _st("c", activate=["s2"]),
                                          _st("d", deactivate=["s2"]), _st("e", activate=["s2"])], P))
    # s1 alone carries P simply supported after the first removal, and every
    # birth after is stress-free: nothing moves the reactions again
    for key in ("RA", "RB"):
        series = np.asarray(out.monitors[key])
        assert series[7] == pytest.approx(5e4, rel=1e-6)
        assert series[-1] == pytest.approx(5e4, rel=1e-6)


def test_a_constraint_kept_after_a_release_is_not_freed_by_it(tmp_path):
    """The support that comes back under B holds it where the release left
    it (1.14: it used to jack B back to the line, and this test compared
    two reactions of that jack), and it is still there in the step after:
    a load put on B then goes straight into it and B does not move."""
    P = [{"region": "M1", "vector": [0, 0, -1e5]}]
    d = _three(tmp_path, "rekeep", [
        _st("a"), _st("b", release=["B"]),
        _st("c", constraints=[{"region": "B", "dofs": [0, 1, 1, 0, 0, 0], "keep": True}]),
        _st("d", loads=[{"region": "B", "vector": [0, 0, -2e5]}])], P)
    d["output"]["monitors"]["wB"] = {"kind": "displacement", "region": "B", "dof": "z"}
    out = _run(d)
    rb, wb = np.asarray(out.monitors["RB"]), np.asarray(out.monitors["wB"])
    assert wb[7] < -1e-4                                    # released, B came down
    assert wb[11] == pytest.approx(wb[7], rel=1e-6) and wb[-1] == pytest.approx(wb[7], rel=1e-6)
    assert abs(rb[11]) < 1e-3 * 2e5                         # holding it there costs nothing
    assert rb[-1] == pytest.approx(2e5, rel=1e-3)           # and what is put on it, it carries


def test_a_birth_under_a_plane_restraint_is_still_on_the_tangent(tmp_path):
    cons = [{"region": "A", "dofs": [1] * 6}, {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}]
    a = _cantilever_with(tmp_path, "plane", cons)
    b = _cantilever_with(tmp_path, "noplane", [{"region": "A", "dofs": [1] * 6}])
    assert a.monitors["wC"][4] == pytest.approx(b.monitors["wC"][4], rel=1e-9)
    assert a.monitors["wC"][4] < -0.01


def _cantilever_with(tmp_path, name, cons):
    d = _case(tmp_path, name, [_st("seg1", activate=["s1"]), _st("seg2", activate=["s2"])],
              constraints=cons)
    d["output"]["monitors"] = {"wC": {"kind": "displacement", "region": "C", "dof": "z"}}
    return _run(d)


def test_a_stay_with_a_designed_prestress_keeps_it(tmp_path):
    d = _case(tmp_path, "stay", [_st("deck"), _st("stay", activate=["cable"])],
              constraints=[{"region": "A", "dofs": [1] * 6}, {"region": "T", "dofs": [1] * 6}])
    d["mesh"]["points"]["T"] = [L, 0, 10.0]
    d["mesh"]["lines"]["cable"] = {"through": ["T", "B"], "size": 20.0}
    d["mesh"]["lines"].pop("s2"); d["mesh"]["points"].pop("C")
    d["materials"]["s"] = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 1.0}
    d["sections"]["cab"] = {"type": "bar", "material": "s", "shape": "general", "area": 1e-3}
    d["assign"] = [{"region": "s1", "section": "g", "orientation": [0, 1, 0]},
                   {"region": "cable", "section": "cab", "prestress": 2.0e8}]
    d["loads"] = []
    d["output"]["monitors"] = {"RT": {"kind": "reaction", "region": "T", "dof": "z"}}
    out = _run(d)
    # tensioned to 200 MPa as it is installed (a strain of 0.01 over 10 m),
    # against the cantilever tip's 3EI/L^3: the two springs in series
    kd = 3 * 3.0e10 * 1.5 ** 3 / 12 / L ** 3
    kc = 2.0e11 * 1e-3 / 10.0
    assert abs(out.monitors["RT"][4]) == pytest.approx(kd * kc * 0.01 / (kd + kc), rel=5e-3)


def test_beta_damping_does_not_see_members_not_built(tmp_path):
    """Rayleigh beta K v on the reference K: a member taken out of the model
    must not damp its nodes (review: 14x overdamped)."""
    damp = {"type": "rayleigh", "alpha": 0.0, "beta": 0.002}
    pulse = [{"region": "B", "vector": [0, 0, -1e6], "amplitude": "kick"}]
    common = {"amplitudes": {"kick": [[0.0, 1.0], [0.05, 1.0], [0.0501, 0.0], [1.0, 0.0]]}}
    cons = [{"region": "A", "dofs": [1] * 6}]
    built = _case(tmp_path, "nb", [{"name": "shake", "deactivate": ["s2"], "loads": pulse}],
                  constraints=cons, loads=[], mode="dynamic",
                  extra=dict(common, solver={"mode": "dynamic", "backend": "numpy", "total_time": 1.0,
                                             "damping": damp}))
    alone = _case(tmp_path, "na", [{"name": "shake", "loads": pulse}], constraints=cons, loads=[],
                  mode="dynamic", extra=dict(common, solver={"mode": "dynamic", "backend": "numpy",
                                                             "total_time": 1.0, "damping": damp}))
    alone["mesh"]["lines"].pop("s2"); alone["mesh"]["points"].pop("C")
    alone["output"]["monitors"] = {"wB": {"kind": "displacement", "region": "B", "dof": "z"}}
    built["output"]["monitors"] = {"wB": {"kind": "displacement", "region": "B", "dof": "z"}}
    a, b = run_case(built), run_case(alone)
    assert a.converged and b.converged
    assert min(a.monitors["wB"]) == pytest.approx(min(b.monitors["wB"]), rel=2e-2)


@pytest.mark.parametrize("steps, words", [
    ([_st("a", birth="design")], "activates none"),
    ([_st("a", release=["B"])], "no inherited constraint"),
])
def test_what_the_review_found_accepted(tmp_path, steps, words):
    cons = [{"region": "A", "dofs": [1, 1, 1, 1, 0, 0]}, {"region": "C", "dofs": [0, 1, 1, 0, 0, 0]}]
    with pytest.raises(ValueError, match=words):
        run_case(_case(tmp_path, "bad", steps, constraints=cons))


# --------------------------------------------------------------------------- #
#  the fibre beam (1.2, item 9 part B)
# --------------------------------------------------------------------------- #

def _fibre(d, kernel=False):
    d["sections"] = {"g": {"type": "fiber", "material": "c", "shape": "rect", "width": 1.0,
                           "height": 1.5, "ny": 2, "nz": 10}}
    if kernel:
        # an elastic steel02 has the kernel's scalar twin; linear_elastic has not
        d["materials"]["c"] = {"type": "steel02", "E": 3.0e10, "Fy": 1.0e12, "b": 0.01, "rho": 2500.0}
    return d


@pytest.mark.parametrize("kernel", [False, True])
def test_a_fibre_girder_cast_span_by_span(tmp_path, kernel):
    got = {}
    out = run_case(_fibre(_case(tmp_path, f"fs{kernel}", [_st("span1", activate=["s1"]),
                                                          _st("span2", activate=["s2"])]), kernel),
                   on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.converged, out.message
    assert (got["s"]._kernel is not None) == kernel
    assert _r(out, "RA") == pytest.approx(7 / 16, rel=2e-3)
    assert _r(out, "RB") == pytest.approx(9 / 8, rel=2e-3)


def test_a_fibre_girder_removes_releases_and_shakes(tmp_path):
    cut = _run(_fibre(_case(tmp_path, "fcut", [_st("both"), _st("cut", deactivate=["s2"])])))
    assert _r(cut, "RA") == pytest.approx(0.5, rel=1e-6) and abs(_r(cut, "RC")) < 1e-9
    still = _run(_fibre(_case(tmp_path, "fstill", [_st("dead"), {"name": "quake", "procedure": "dynamic",
                                                                  "total_time": 0.2}])))
    w = np.asarray(still.monitors["w"])
    # the dead state is an equilibrium to the relaxation's tolerance (1e-7):
    # released, a state off by r swings r either side of the equilibrium, so
    # the peak to peak is up to twice it. It was 8e-9 of w until the fibre
    # solver rebuilt its relaxation mass at each level, and
    # is 1.6e-7 since: the relaxation stops at another point inside its
    # tolerance.
    assert np.ptp(w[4:]) < 2e-7 * abs(w[3])
    static = _run(_fibre(_case(tmp_path, "fstat", [{"name": "span1", "activate": ["s1"], "load_steps": 4,
                                                     "procedure": "static"},
                                                    {"name": "span2", "activate": ["s2"], "load_steps": 4,
                                                     "procedure": "static"}])))
    assert _r(static, "RA") == pytest.approx(7 / 16, rel=2e-3)
    with h5py.File(cut.result_path, "r") as f:
        frames = sorted(f["steps"], key=int)
        last = f["steps"][frames[-1]]
        assert np.asarray(last["member_built"]).sum() == 10
        # and a member not built has not failed: no failure field is written at all
        assert "member_failed" not in last


# --------------------------------------------------------------------------- #
#  each segment's own concrete age (1.2, item 9 part C)
# --------------------------------------------------------------------------- #

def test_a_segment_born_late_creeps_on_its_own_clock(tmp_path):
    """Two independent bars under a held axial force. The first is loaded at
    28 days; the second is cast later, born when the model's clock reads 100
    days at its own age of 7 days, and loaded then. At 1000 days each has
    stretched N/EA (1 + phi(t, t0)) on ITS clock: phi(1000, 28) and
    phi(907, 7)."""
    from vfife.core.creep import creep_coefficient
    spec = {"model": "aci209"}
    N, E, A, Lb = 1.0e6, 3.0e10, 0.25, 10.0
    d = {"version": 1, "name": "age", "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [Lb, 0, 0], "C": [0, 5, 0], "D": [Lb, 5, 0]},
                  "lines": {"b1": {"through": ["A", "B"], "size": 2.5},
                            "b2": {"through": ["C", "D"], "size": 2.5}}},
         "materials": {"c": {"type": "linear_elastic", "E": E, "nu": 0.2, "rho": 2500.0, "creep": spec}},
         "sections": {"g": {"type": "beam", "material": "c", "shape": "rect", "width": 0.5, "height": 0.5}},
         "assign": [{"region": "all", "section": "g", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "C", "dofs": [1] * 6},
                         {"region": "all", "dofs": [0, 1, 1, 1, 1, 1]}],
         "loads": [],
         "steps": [
             {"name": "load1", "load_steps": 1, "max_steps_per_load": 40000, "activate": ["b1"],
              "loads": [{"region": "B", "vector": [N, 0, 0], "keep": True}]},
             {"name": "creep1", "age": 28.0, "to_age": 100.0, "load_steps": 12, "max_steps_per_load": 40000},
             {"name": "cast2", "activate": ["b2"], "activate_age": 7.0, "load_steps": 1,
              "max_steps_per_load": 40000,
              "loads": [{"region": "D", "vector": [N, 0, 0], "keep": True}]},
             {"name": "creep2", "to_age": 1000.0, "load_steps": 24, "max_steps_per_load": 40000}],
         "solver": {"mode": "relaxation", "backend": "numpy", "tolerance": 1e-8, "age": 28.0},
         "output": {"path": str(tmp_path / "age.h5"), "monitors": {
             "uB": {"kind": "displacement", "region": "B", "dof": "x"},
             "uD": {"kind": "displacement", "region": "D", "dof": "x"}}}}
    out = _run(d)
    e0 = N * Lb / (E * A)
    assert out.monitors["uB"][-1] / e0 == pytest.approx(1 + float(creep_coefficient(spec, 1000.0, 28.0)), rel=5e-3)
    assert out.monitors["uD"][-1] / e0 == pytest.approx(1 + float(creep_coefficient(spec, 907.0, 7.0)), rel=5e-3)
    # and the late bar's stretch at its birth was elastic only
    i = 1 + 12 + 1 - 1
    assert out.monitors["uD"][i] / e0 == pytest.approx(1.0, rel=1e-4)


def test_a_segment_born_late_shrinks_from_its_birth(tmp_path):
    """A free bar cast at clock 100 at age 7 shortens by
    (eps_sh(907) - eps_sh(7)) L by clock 1000; its shrinkage in the formwork
    is not a load."""
    from vfife.core.creep import shrinkage_strain
    spec = {"model": "aci209"}
    Lb = 10.0
    d = {"version": 1, "name": "sh", "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [Lb, 0, 0], "C": [0, 5, 0], "D": [Lb, 5, 0]},
                  "lines": {"b1": {"through": ["A", "B"], "size": 2.5},
                            "b2": {"through": ["C", "D"], "size": 2.5}}},
         "materials": {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2500.0,
                             "shrinkage": spec}},
         "sections": {"g": {"type": "beam", "material": "c", "shape": "rect", "width": 0.5, "height": 0.5}},
         "assign": [{"region": "all", "section": "g", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "C", "dofs": [1] * 6},
                         {"region": "all", "dofs": [0, 1, 1, 1, 1, 1]}],
         "loads": [],
         "steps": [
             {"name": "s0", "age": 28.0, "to_age": 100.0, "load_steps": 4, "max_steps_per_load": 40000},
             {"name": "cast2", "activate": ["b2"], "activate_age": 7.0, "load_steps": 1, "max_steps_per_load": 40000},
             {"name": "s1", "to_age": 1000.0, "load_steps": 6, "max_steps_per_load": 40000}],
         "solver": {"mode": "relaxation", "backend": "numpy", "tolerance": 1e-9, "age": 28.0},
         "output": {"path": str(tmp_path / "sh.h5"), "monitors": {
             "uB": {"kind": "displacement", "region": "B", "dof": "x"},
             "uD": {"kind": "displacement", "region": "D", "dof": "x"}}}}
    out = _run(d)
    assert out.monitors["uD"][-1] == pytest.approx(
        Lb * (shrinkage_strain(spec, 907.0) - shrinkage_strain(spec, 7.0)), rel=1e-4)
    assert out.monitors["uD"][4] == pytest.approx(0.0, abs=1e-9)        # born free, where it was cast
    assert out.monitors["uB"][-1] == pytest.approx(Lb * shrinkage_strain(spec, 1000.0), rel=1e-4)


def _held_bar(tmp_path, name, steps, age0):
    return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [10.0, 0, 0], "C": [0, 5, 0], "D": [10.0, 5, 0]},
                     "lines": {"b1": {"through": ["A", "B"], "size": 2.5},
                               "b2": {"through": ["C", "D"], "size": 2.5}}},
            "materials": {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2500.0,
                                "creep": {"model": "aci209"}}},
            "sections": {"g": {"type": "beam", "material": "c", "shape": "rect", "width": 0.5, "height": 0.5}},
            "assign": [{"region": "all", "section": "g", "orientation": [0, 0, 1]}],
            "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "C", "dofs": [1] * 6},
                            {"region": "all", "dofs": [0, 1, 1, 1, 1, 1]}],
            "loads": [], "steps": steps,
            "solver": {"mode": "relaxation", "backend": "numpy", "tolerance": 1e-9, "age": age0},
            "output": {"path": str(tmp_path / f"{name}.h5"), "monitors": {
                "RD": {"kind": "reaction", "region": "D", "dof": "x"}}}}


def test_a_late_segment_s_relaxation_is_the_one_of_a_bar_loaded_at_its_age(tmp_path):
    """A held stretch relaxes: the born bar's force over its own 7 -> 907 days
    is the force of a bar loaded at 7 days and held to 907, station for
    station -- the increments filed where they were dated (review, 1.2: 10%
    off on twelve linear stations)."""
    hold = {"region": "D", "dofs": [1, 0, 0, 0, 0, 0], "value": 1.0e-3, "keep": True}
    born = _run(_held_bar(tmp_path, "born", [
        {"name": "s0", "age": 28.0, "to_age": 100.0, "load_steps": 4, "max_steps_per_load": 40000},
        {"name": "cast", "activate": ["b2"], "activate_age": 7.0, "load_steps": 1,
         "max_steps_per_load": 40000, "constraints": [hold]},
        {"name": "relax", "to_age": 1000.0, "load_steps": 12, "time_spacing": "linear",
         "max_steps_per_load": 40000}], 28.0))
    ref = _run(_held_bar(tmp_path, "ref", [
        {"name": "load", "load_steps": 1, "max_steps_per_load": 40000, "constraints": [hold],
         "deactivate": ["b1"]},
        {"name": "relax", "age": 7.0, "to_age": 907.0, "load_steps": 12, "time_spacing": "linear",
         "max_steps_per_load": 40000}], 7.0))
    b, r = np.asarray(born.monitors["RD"][-12:]), np.asarray(ref.monitors["RD"][-12:])
    assert b == pytest.approx(r, rel=1e-6)


def test_a_segment_born_at_the_clock_s_own_age_is_still_born(tmp_path):
    """shift 0 is still a birth: nothing dated before it (review, 1.2)."""
    from vfife.core.creep import creep_coefficient
    load = {"region": "D", "vector": [1.0e6, 0, 0], "keep": True}
    d = _held_bar(tmp_path, "s0", [
        {"name": "c1", "age": 28.0, "to_age": 100.0, "load_steps": 4, "max_steps_per_load": 40000},
        {"name": "cast", "age": 150.0, "activate": ["b2"], "activate_age": 150.0, "load_steps": 1,
         "max_steps_per_load": 40000, "loads": [load]},
        {"name": "c2", "to_age": 1000.0, "load_steps": 24, "max_steps_per_load": 40000}], 28.0)
    d["output"]["monitors"] = {"uD": {"kind": "displacement", "region": "D", "dof": "x"}}
    out = _run(d)
    e0 = 1.0e6 * 10.0 / (3.0e10 * 0.25)
    assert out.monitors["uD"][4] / e0 == pytest.approx(1.0, rel=1e-4)
    assert out.monitors["uD"][-1] / e0 == pytest.approx(1 + float(creep_coefficient({"model": "aci209"}, 1000.0, 150.0)),
                                                        rel=5e-3)


def _rc(tmp_path, name, steps):
    return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [2.0, 0, 0], "C": [4.0, 0, 0]},
                     "lines": {"s1": {"through": ["A", "B"], "size": 0.5}, "s2": {"through": ["B", "C"], "size": 0.5}}},
            "materials": {"concrete": {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6,
                                       "epscu": 0.005, "ft": 2.0e6, "Ets": 1.0e9, "rat": 0.1, "rho": 2400},
                          "steel": {"type": "steel02", "E": 2.0e11, "Fy": 4.0e8, "b": 0.01, "R0": 20.0, "rho": 7850}},
            "sections": {"no20": {"type": "bar", "material": "steel", "shape": "general", "area": 3.14e-4},
                         "g": {"type": "fiber", "material": "concrete", "shape": "rect", "width": 0.3, "height": 0.4,
                               "ny": 2, "nz": 12,
                               "bars": [{"y": 0.1, "z": 0.15, "section": "no20"}, {"y": -0.1, "z": 0.15, "section": "no20"},
                                        {"y": 0.1, "z": -0.15, "section": "no20"},
                                        {"y": -0.1, "z": -0.15, "section": "no20"}]}},
            "assign": [{"region": "all", "section": "g", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
            "loads": [], "steps": steps,
            "solver": {"mode": "relaxation", "backend": "numpy", "tolerance": 1e-5, "n_steps": 100},
            "output": {"path": str(tmp_path / f"{name}.h5"), "fields": ["displacement"], "monitors": {
                "N2": {"kind": "element", "region": "s2", "field": "axial_force", "reduce": "absmax"}}}}


def test_a_fibre_member_built_again_is_new_concrete(tmp_path):
    """Yielded, removed and built again: its fibres start from their first
    state, not from the cracked and yielded one (review, 1.2)."""
    got = {}
    out = run_case(_rc(tmp_path, "reb", [
        {"name": "one", "load_steps": 6, "max_steps_per_load": 40000,
         "loads": [{"region": "C", "vector": [0, 0, -3.5e4]}]},
        {"name": "cut", "deactivate": ["s2"], "load_steps": 2, "max_steps_per_load": 40000},
        {"name": "reb", "activate": ["s2"], "load_steps": 2, "max_steps_per_load": 40000}]),
        on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.converged, out.message
    assert abs(out.monitors["N2"][5]) > 1e4                    # it carried the load
    assert abs(out.monitors["N2"][-1]) < 1e-3                  # born again, unstressed


def test_a_connector_born_late_in_a_fibre_model_is_refused(tmp_path):
    d = _rc(tmp_path, "conn", [{"name": "a", "load_steps": 1, "max_steps_per_load": 1000},
                               {"name": "b", "activate": ["link"], "load_steps": 1, "max_steps_per_load": 1000}])
    d["mesh"]["points"]["E"] = [4.0, 0, -0.5]
    d["mesh"]["lines"]["link"] = {"through": ["C", "E"], "size": 1.0}
    d["sections"]["brg"] = {"type": "connector", "stiffness": [1e8] * 6}
    d["assign"] = [{"region": "s1", "section": "g", "orientation": [0, 1, 0]},
                   {"region": "s2", "section": "g", "orientation": [0, 1, 0]},
                   {"region": "link", "section": "brg", "orientation": [0, 1, 0]}]
    d["constraints"].append({"region": "E", "dofs": [1] * 6})
    with pytest.raises(ValueError, match="connector"):
        run_case(d)


@pytest.mark.parametrize("cable", [{"prestress": 1.0e6}, {"tension_only": True}])
def test_a_segment_reached_through_a_stay_cut_to_length_is_born_on_the_deck(tmp_path, cable):
    """A stay is not a rigid arm: a new segment that a stay from a standing
    tower reaches as soon as the deck does is born on the deck's tangent,
    as it is without the stay (a jacked stay and a
    connector were excluded in 1.13, a stay cut to length was not)."""
    def tip(with_stay, name):
        d = _case(tmp_path, name, [_st("seg1", activate=["s1"] + (["tower"] if with_stay else [])),
                                   _st("seg2", activate=["s2"] + (["cable"] if with_stay else []))],
                  constraints=[{"region": "A", "dofs": [1] * 6}] + ([{"region": "G", "dofs": [1] * 6}] if with_stay else []),
                  mode="static")
        d["output"]["monitors"] = {"wC": {"kind": "displacement", "region": "C", "dof": "z"}}
        if with_stay:
            # a tower standing from the first step: the stay's top is a standing node, one hop from C
            d["mesh"]["points"]["G"] = [2 * L + 5.0, 0, -30.0]
            d["mesh"]["points"]["T"] = [2 * L + 5.0, 0, 30.0]
            d["mesh"]["lines"]["tower"] = {"through": ["G", "T"], "size": 60.0}
            d["mesh"]["lines"]["cable"] = {"through": ["T", "C"], "size": 40.0}
            d["materials"]["s"] = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 1.0e-6}
            d["sections"]["cab"] = {"type": "bar", "material": "s", "shape": "general", "area": 1.0e-9}
            d["assign"] = [{"region": "s1", "section": "g", "orientation": [0, 1, 0]},
                           {"region": "s2", "section": "g", "orientation": [0, 1, 0]},
                           {"region": "tower", "section": "g", "orientation": [0, 1, 0]},
                           dict({"region": "cable", "section": "cab"}, **cable)]
        return _run(d).monitors["wC"][-1]
    free = tip(False, "free")
    assert free < -0.01
    assert tip(True, "stay") == pytest.approx(free, rel=1e-4)


def test_a_stay_meshed_in_several_bars_is_born_straight_between_its_ends():
    """The new nodes inside a stay that places nothing go between where its
    ends are, along its length (review): the tower top T
    has moved by (0, 0, -0.1), the deck end C was put at (0, 0, -0.7); the
    middle nodes follow in proportion, not stay at their design positions
    and kink the stay into its last bar. A stay that touches nothing placed
    or standing keeps its design position."""
    from types import SimpleNamespace
    from vfife.core.analysis_steps import _between_cable_ends
    # T=0 (standing), 1 and 2 inside, C=3 (placed); 4-5 a stay on its own
    X = np.array([[0, 0, 30.0], [10, 0, 20.0], [30, 0, 0.0], [40, 0, -10.0], [0, 5, 0.0], [5, 5, 0.0]])
    elements = np.array([[0, 1], [1, 2], [2, 3], [4, 5]])
    solver = SimpleNamespace(mesh=SimpleNamespace(nodes=X, elements=elements))
    qv = np.zeros((6, 6))
    qv[0, 2] = -0.1
    qv[3, 2] = -0.7
    _between_cable_ends(solver, np.arange(4), {1, 2, 3, 4, 5}, {3: 3}, qv)
    s = np.array([0.0, 1.0, 3.0, 4.0]) / 4.0  # arc length along T-1-2-C
    assert qv[[0, 1, 2, 3], 2] == pytest.approx(-0.1 - 0.6 * s, abs=1e-9)
    assert qv[[1, 2], :2] == pytest.approx(np.zeros((2, 2)), abs=1e-12)
    assert qv[[4, 5], :3] == pytest.approx(np.zeros((2, 3)), abs=1e-9)
