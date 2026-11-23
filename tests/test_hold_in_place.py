"""A support put under a node that has moved holds it where it stands (1.14). A cantilever is loaded, a support is then put under its tip,
and it is loaded again: the support takes the second load and none of the
first. It used to pull the tip back to the line it was drawn on."""
import sys
from pathlib import Path

import numpy as np
import pytest

from vfife.api import run_case

E, I, L, P = 2.0e11, 1.0e-4, 4.0, 1.0e4
KB = 3 * E * I / L ** 3
W1 = -P / KB                         # the tip under the first load
T1 = P * L ** 2 / (2 * E * I)        # and its rotation (about y, downward tip: positive)


def _case(tmp_path, steps, mode="static", backend="numpy", name="hold", fibre=False):
    sol = {"mode": mode, "backend": backend, "tolerance": 1e-9 if mode == "static" else 1e-7}
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [L, 0, 0], "C": [2 * L, 0, 0]},
                  "lines": {"beam": {"through": ["A", "B"], "size": 1.0}, "next": {"through": ["B", "C"], "size": 4.0}}},
         "materials": {"steel": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 1.0}},
         "sections": {"s": {"type": "beam", "material": "steel", "shape": "general",
                            "area": 0.01, "iy": I, "iz": I, "j": 2 * I}},
         "assign": [{"region": "all", "section": "s", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "A", "dofs": [1] * 6}], "loads": [], "steps": steps, "solver": sol,
         "output": {"path": str(tmp_path / f"{name}.h5"),
                    "monitors": {"w": {"kind": "displacement", "region": "B", "dof": "z"},
                                 "R": {"kind": "reaction", "region": "B", "dof": "z"},
                                 "RC": {"kind": "reaction", "region": "C", "dof": "z"}}}}
    if fibre:
        d["sections"] = {"s": {"type": "fiber", "material": "steel", "shape": "rect",
                               "width": 0.1, "height": (12 * I / 0.1) ** (1 / 3), "ny": 2, "nz": 12}}
    return d


def _st(name, mode, **kw):
    st = dict({"name": name}, **kw)
    if mode == "relaxation":
        st.update({"load_steps": 2, "max_steps_per_load": 200000})
    return st


def _load(p):
    return {"region": "B", "vector": [0, 0, -p], "keep": True}


def _steps(mode, support, first=None):
    """The second span is never built: the cantilever alone."""
    return [_st("load", mode, activate=["beam"], loads=[_load(P)], **(first or {})),
            _st("prop", mode, constraints=[dict(support, region="B", keep=True)]),
            _st("more", mode, loads=[_load(2 * P)])]


def _run(tmp_path, steps, **kw):
    got = {}
    out = run_case(_case(tmp_path, steps, **kw), on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status == "CONVERGED", out.message
    return out, got["s"]


@pytest.mark.parametrize("mode,backend", [("static", "numpy"), ("static", "numba"),
                                          ("relaxation", "numpy"), ("relaxation", "numba")])
def test_a_support_put_under_a_loaded_tip_holds_it_where_it_is(tmp_path, mode, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    out, s = _run(tmp_path, _steps(mode, {"dofs": [0, 0, 1, 0, 0, 0]}), mode=mode, backend=backend)
    w, R = np.asarray(out.monitors["w"]), np.asarray(out.monitors["R"])
    assert w[-1] == pytest.approx(W1, rel=2e-4)          # where the first load left it
    assert R[-1] == pytest.approx(2 * P, rel=1e-6)       # and the support carries the second load alone
    # nothing moved after the support came: it was W1 at every record from then on
    held = np.flatnonzero(np.abs(R) > 1e-3 * P)
    assert np.allclose(w[held[0] - 1:], W1, rtol=2e-4)


def test_a_value_brings_the_support_to_a_position(tmp_path):
    """To jack the tip back to the line is to say so: `value: 0`."""
    out, s = _run(tmp_path, _steps("static", {"dofs": [0, 0, 1, 0, 0, 0], "value": 0.0}))
    w, R = np.asarray(out.monitors["w"]), np.asarray(out.monitors["R"])
    assert abs(w[-1]) < 1e-12
    # the jack carries what the propped cantilever's support does: P and 2P together
    assert R[-1] == pytest.approx(3 * P, rel=1e-4)


def test_a_clamp_put_on_a_turned_tip_holds_its_rotation(tmp_path):
    out, s = _run(tmp_path, _steps("static", {"dofs": [1] * 6}))
    n = int(s._region("B")[0])
    qv = s.dofs.view(s.q)
    assert qv[n, 2] == pytest.approx(W1, rel=2e-4) and qv[n, 4] == pytest.approx(T1, rel=2e-4)
    assert np.allclose(s.orient[n] @ s.orient[n].T, np.eye(3), atol=1e-14)
    assert abs(s.orient[n][0, 2]) == pytest.approx(np.sin(T1), rel=2e-4)     # the triad is turned, not the identity
    assert np.asarray(out.monitors["R"])[-1] == pytest.approx(2 * P, rel=1e-6)


def test_a_dof_held_from_the_start_is_at_zero_and_one_a_value_moved_returns(tmp_path):
    """What does not change. The root never moves; and a tip pushed by a
    `value` and then taken by a plain constraint was never free: it goes
    back to zero, as it always did."""
    steps = [_st("push", "static", activate=["beam"],
                 constraints=[{"region": "B", "dofs": [0, 0, 1, 0, 0, 0], "value": -0.01}]),
             _st("back", "static", constraints=[{"region": "B", "dofs": [0, 0, 1, 0, 0, 0], "keep": True}])]
    out, s = _run(tmp_path, steps)
    w = np.asarray(out.monitors["w"])
    assert w.min() == pytest.approx(-0.01, rel=1e-9) and abs(w[-1]) < 1e-12
    assert not np.abs(s.q[s.dofs.node_dofs(s._region("A"), s.dofs.components).ravel()]).any()


def test_a_released_support_lets_go_and_the_next_one_holds_where_it_finds_it(tmp_path):
    steps = _steps("static", {"dofs": [0, 0, 1, 0, 0, 0]}) + [
        _st("free", "static", release=["B"]),
        _st("again", "static", constraints=[{"region": "B", "dofs": [0, 0, 1, 0, 0, 0], "keep": True}]),
        _st("off", "static", loads=[_load(-3 * P)])]
    out, s = _run(tmp_path, steps)
    w, R = np.asarray(out.monitors["w"]), np.asarray(out.monitors["R"])
    # freed under 3P it goes to 3 W1; held there; unloaded, the support pulls it down with 3P
    assert w[-1] == pytest.approx(3 * W1, rel=2e-4)
    assert R[-1] == pytest.approx(-3 * P, rel=1e-4)


def test_a_dynamic_step_holds_it_too(tmp_path):
    steps = [dict(st, procedure="static") for st in _steps("static", {"dofs": [0, 0, 1, 0, 0, 0]})]
    steps[2] = {"name": "more", "procedure": "dynamic", "total_time": 0.02, "loads": [_load(2 * P)]}
    d = _case(tmp_path, steps, mode="dynamic")
    d["solver"].update({"total_time": 0.02, "damping": {"type": "viscous", "coeff": 50.0}})
    out = run_case(d)
    w = np.asarray(out.monitors["w"])
    assert w[-1] == pytest.approx(W1, rel=2e-4) and np.ptp(w[-20:]) < 1e-12


def test_the_fibre_beam_holds_it(tmp_path):
    out, s = _run(tmp_path, _steps("static", {"dofs": [0, 0, 1, 0, 0, 0]}), fibre=True)
    w, R = np.asarray(out.monitors["w"]), np.asarray(out.monitors["R"])
    held = np.flatnonzero(np.abs(R) > 1e-3 * P)
    assert abs(w[held[0] - 1]) > 0.5 * abs(W1)           # it had come down
    assert np.allclose(w[held[0]:], w[held[0] - 1], rtol=1e-9)
    assert R[-1] == pytest.approx(2 * P, rel=1e-5)


def test_a_run_stopped_while_a_support_holds_resumes(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_restart_resumes import _same, _split
    d = _case(tmp_path, _steps("relaxation", {"dofs": [1] * 6}), mode="relaxation")
    d.pop("name"); d["output"].pop("path")
    for where in (1, 2):                                 # in the step that takes it, and in the one after
        sub = tmp_path / f"at{where}"; sub.mkdir()
        full, half, rest = _split(sub, d, lambda s, stage, k=where: s.step_index == k and stage >= 1,
                                  progress_every=20)
        _same(full, rest)
        n = int(rest._region("B")[0])
        assert rest.dofs.view(rest.q)[n, 2] == pytest.approx(W1, rel=1e-3)


def test_a_one_step_run_begun_from_a_held_state_keeps_the_hold_through_a_restart(tmp_path):
    """Found in review: the hold came in with `initial_state` and was not in
    the one-step run's checkpoint; restarted, the tip snapped to zero."""
    from vfife.core.diagnostics import AnalysisInterrupted
    d0 = _case(tmp_path, _steps("static", {"dofs": [0, 0, 1, 0, 0, 0]}), name="first")
    assert run_case(d0).status == "CONVERGED"
    d = _case(tmp_path, [], mode="dynamic", name="then")
    d.pop("steps")
    d["constraints"] = [{"region": "A", "dofs": [1] * 6}, {"region": "B", "dofs": [0, 0, 1, 0, 0, 0]},
                        {"region": "C", "dofs": [1] * 6}]
    d["loads"] = [{"region": "B", "vector": [0, 0, -3 * P], "amplitude": "hold"}]
    d["solver"].update({"total_time": 0.004, "damping": {"type": "viscous", "coeff": 50.0},
                        "initial_state": str(tmp_path / "first.h5")})
    got = {}
    run_case(dict(d, output=dict(d["output"], path=str(tmp_path / "full.h5"))), progress_every=10,
             on_ready=lambda s, _p: got.__setitem__("full", s))
    with pytest.raises(AnalysisInterrupted):
        run_case(dict(d, output=dict(d["output"], path=str(tmp_path / "half.h5"))), progress_every=10,
                 on_ready=lambda s, _p: got.__setitem__("half", s), on_progress=lambda local, *_: local < 20)
    run_case(dict(d, output=dict(d["output"], path=str(tmp_path / "rest.h5"))), restart=str(tmp_path / "half.h5"),
             progress_every=10, on_ready=lambda s, _p: got.__setitem__("rest", s))
    n = int(got["rest"]._region("B")[0])
    for s in (got["full"], got["rest"]):
        assert s.dofs.view(s.q)[n, 2] == pytest.approx(W1, rel=2e-4)
    assert np.array_equal(got["full"].q, got["rest"].q)


def test_a_span_built_again_under_a_support_that_held_it_is_born_at_the_support(tmp_path):
    """Found in review: the far end C was held where it stood, the span was
    removed and built again -- at zero, and then dragged to the old held
    value, the new span stressed by the way. A node built again is a new
    node: the support is where it is."""
    st = lambda name, **kw: _st(name, "static", **kw)
    steps = [st("load", activate=["beam", "next"], loads=[_load(P)]),
             st("hold", constraints=[{"region": "C", "dofs": [1, 1, 1, 0, 0, 0], "keep": True}]),
             st("out", deactivate=["next"]), st("in", activate=["next"]), st("idle")]
    out, s = _run(tmp_path, steps)
    e = int(s._members_of("next")[0])
    assert np.abs(s.end_actions[e]).max() < 1e-6 * P * L
    assert abs(np.asarray(out.monitors["RC"])[-1]) < 1e-6 * P
    assert np.asarray(out.monitors["w"])[-1] == pytest.approx(W1, rel=2e-4)


# --------------------------------------------------------------------------- #
#  a new member's end at a clamped support
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("birth", ["tangent", "design"])
def test_a_new_span_to_a_clamped_support_is_born_without_stress(tmp_path, birth):
    """The cantilever's tip has come down and turned; the next span is then
    built from it to a clamped support. It is born between the tip as it
    stands and the support as it is, free of stress: the support feels
    nothing until something is added. Its end used to be born with the tip's
    rotation and turned back by the clamp over the step."""
    cons = [{"region": "A", "dofs": [1] * 6}, {"region": "C", "dofs": [1] * 6}]
    steps = [_st("load", "static", activate=["beam"], loads=[_load(P)]),
             _st("span", "static", activate=["next"], birth=birth)]
    d = _case(tmp_path, steps); d["constraints"] = cons
    got = {}
    out = run_case(d, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status == "CONVERGED", out.message
    s = got["s"]
    w = np.asarray(out.monitors["w"])
    assert w[-1] == pytest.approx(W1, rel=2e-4)                       # the tip did not move
    assert abs(np.asarray(out.monitors["RC"])[-1]) < 1e-6 * P         # and the support carries nothing
    e = int(s._members_of("next")[0])
    assert np.abs(s.end_actions[e]).max() < 1e-6 * P * L             # no moment either: nothing in the span
    n = int(s._region("C")[0])
    assert np.array_equal(s.orient[n], np.eye(3)) and not s.dofs.view(s.q)[n].any()
