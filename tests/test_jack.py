"""A cable stressed to a FORCE (1.13): `jack: T` on a bar.
In the step that builds it the jack holds T times the step's progress,
whatever the structure gives; at the step's end the bar carries exactly T
and keeps the unstressed length that gives it. `prestress` is the other
statement -- a length -- and loses force as the structure comes to it.

A cantilever's tip is hung from an inclined stay (3-4-5) to a fixed point."""
import sys
from pathlib import Path

import numpy as np
import pytest

from vfife.api import run_case

E, I, L, A, T = 2.0e11, 1.0e-4, 4.0, 1.0e-4, 5.0e4
KB = 3 * E * I / L ** 3


def _case(tmp_path, steps, stay, backend="numpy"):
    return {
        "version": 1, "name": "jack", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [L, 0, 0], "T": [0, 0, 3.0]},
                 "lines": {"beam": {"through": ["A", "B"], "size": 1.0}, "stay": {"through": ["T", "B"], "size": 9.0}}},
        "materials": {"steel": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 1.0}},
        "sections": {"s": {"type": "beam", "material": "steel", "shape": "general",
                           "area": 0.01, "iy": I, "iz": I, "j": 2 * I},
                     "c": {"type": "bar", "material": "steel", "shape": "general", "area": A}},
        "assign": [{"region": "beam", "section": "s", "orientation": [0, 1, 0]},
                   dict({"region": "stay", "section": "c"}, **stay)],
        "constraints": [{"region": "A", "dofs": [1, 1, 1, 1, 1, 1]}, {"region": "T", "dofs": [1, 1, 1, 0, 0, 0]}],
        "loads": [], "steps": steps, "solver": {"mode": "static", "backend": backend, "tolerance": 1e-9},
        "output": {"path": str(tmp_path / "jack.h5"),
                   "monitors": {"w": {"kind": "displacement", "region": "B", "dof": "z"}}},
    }


def _load(p):
    return {"region": "B", "vector": [0, 0, -p], "keep": True}


def _run(tmp_path, steps, stay, backend="numpy"):
    got = {}
    out = run_case(_case(tmp_path, steps, stay, backend), on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status == "CONVERGED", out.message
    s = got["s"]
    e = int(np.flatnonzero(np.asarray(s.member_kind) == "truss2")[0])
    return out, s, float(s.end_actions[e, 6])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_at_the_end_of_its_step_the_stay_carries_the_jack_force(tmp_path, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    out, s, N = _run(tmp_path, [{"name": "stress"}], {"jack": T}, backend)
    assert N == pytest.approx(T, rel=1e-9)
    w = float(np.asarray(out.monitors["w"])[-1])
    assert w == pytest.approx(0.6 * T / KB, rel=1e-2)                  # the tip rises by the stay's lift
    # a prestress of the same force is the length that gives it in the MESH: the tip comes up and it is lost
    _o, _s, Np = _run(tmp_path, [{"name": "stress"}], {"prestress": T / A}, backend)
    assert Np < 0.5 * T


def test_halfway_through_its_step_the_jack_holds_half(tmp_path):
    seen = []
    got = {}

    def prog(count, *_):
        s = got["s"]
        e = int(np.flatnonzero(np.asarray(s.member_kind) == "truss2")[0])
        seen.append(float(s.end_actions[e, 6]))
        return True
    run_case(_case(tmp_path, [{"name": "stress"}], {"jack": T}), on_ready=lambda s, _p: got.__setitem__("s", s),
             on_progress=prog, progress_every=1)
    assert seen[4] == pytest.approx(0.5 * T, rel=1e-9) and seen[-1] == pytest.approx(T, rel=1e-9)


def test_after_its_step_it_is_an_ordinary_bar(tmp_path):
    out, s, N = _run(tmp_path, [{"name": "stress"}, {"name": "load", "loads": [_load(2.0e4)]}], {"jack": T})
    w = np.asarray(out.monitors["w"])
    k_stay = E * A / 5.0 * 0.36                                        # its vertical stiffness at the tip
    assert w[-1] - w[9] == pytest.approx(-2.0e4 / (KB + k_stay), rel=2e-2)
    assert N == pytest.approx(T + 2.0e4 * k_stay / (KB + k_stay) / 0.6, rel=2e-2)
    assert s._jack_done.any() and N > 1.3 * T


def test_a_stay_built_later_is_jacked_in_the_step_that_builds_it(tmp_path):
    steps = [{"name": "load", "loads": [_load(2.0e4)]}, {"name": "stay", "activate": ["stay"]}]
    out, s, N = _run(tmp_path, steps, {"jack": T})
    w = np.asarray(out.monitors["w"])
    assert N == pytest.approx(T, rel=1e-9)
    assert w[9] == pytest.approx(-2.0e4 / KB, rel=1e-3) and w[-1] - w[9] == pytest.approx(0.6 * T / KB, rel=2e-2)


def test_a_run_stopped_in_and_after_the_jacking_step_resumes(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_restart_resumes import _same, _split
    for where in (0, 1):
        d = _case(tmp_path, [{"name": "stress"}, {"name": "load", "loads": [_load(2.0e4)]}], {"jack": T})
        d.pop("name"); d["output"].pop("path")
        sub = tmp_path / f"r{where}"; sub.mkdir()
        full, half, rest = _split(sub, d, lambda s, level, where=where: s.step_index == where and level >= 4,
                                  progress_every=1)
        _same(full, rest)


def test_jack_is_one_statement_of_a_cables_length_and_the_frame_models(tmp_path):
    with pytest.raises(Exception, match="ONE of"):
        run_case(_case(tmp_path, [{"name": "a"}], {"jack": T, "prestress": 1e8}))
    # a model of cables alone is a frame model too (the mixed line solver); a fibre model is refused by name
    from vfife.core.structural import _refuse_jack
    _refuse_jack({"members": [{"jack": T}], "element": {"type": "mixed_line"}}, "mixed_line")
    with pytest.raises(ValueError, match="this is a beam2_fiber model"):
        _refuse_jack({"members": [{"jack": T}], "element": {"type": "beam2_fiber"}}, "beam2_fiber")


# ---- found in review ---------------------------------------

def test_outside_a_run_the_cable_is_the_bar_it_is(tmp_path):
    """A modal analysis of the model as built: the jack lines made the cable
    follow every perturbation of the tangent with no force, and its
    stiffness was missing -- 16.77 Hz with the stay and without."""
    from vfife.core import analysis_steps as st
    got = {}
    d = _case(tmp_path, [{"name": "stress"}], {"jack": T})
    run_case(d, on_ready=lambda s, _p: got.__setitem__("s", s))
    s = got["s"]
    assert not s._jack_live                       # the run is over
    s._jack_done[:] = False; s.step_index = 0; s._jack_p = 0.5
    before = s.member_Lu.copy()
    st.jack_lengths(s)
    assert np.array_equal(s.member_Lu, before)    # nothing is jacked when no step is in progress


def test_a_bar_built_from_the_start_is_jacked_in_the_first_step_whatever_comes_after(tmp_path):
    steps = [{"name": "stress"}, {"name": "out", "deactivate": ["stay"]}, {"name": "again", "activate": ["stay"]}]
    got = {}
    run_case(_case(tmp_path, steps, {"jack": T}), on_ready=lambda s, _p: got.__setitem__("s", s))
    s = got["s"]
    e = int(np.flatnonzero(np.asarray(s.member_kind) == "truss2")[0])
    assert s._jack_step_of[e] == 0                # it was 2: unstressed through the first step


def test_a_jack_is_not_worked_in_a_dynamic_step(tmp_path):
    d = _case(tmp_path, [{"name": "shake", "procedure": "dynamic", "total_time": 0.01}], {"jack": T})
    d["solver"] = {"mode": "dynamic", "backend": "numpy", "total_time": 0.01}
    out = None
    with pytest.raises(Exception, match="a jack is worked in a static or a relaxation step"):
        out = run_case(d)
        raise AssertionError(out.message)


def test_a_cable_being_jacked_and_a_connector_carry_no_new_node(tmp_path):
    """Under `birth: tangent` a new node is carried rigidly by the standing
    node it is reached from first. Through a cable that put a rib segment
    where the tower top's rotation sent it (0.27 m on the real arch): a new
    node reached only through a jacked stay stays where the model has it."""
    got = {}
    d = _case(tmp_path, [{"name": "load", "loads": [_load(2.0e4)]}, {"name": "stay", "activate": ["stay"]}], {"jack": T})
    d["constraints"] = [{"region": "A", "dofs": [1, 1, 1, 1, 1, 1]}]        # the stay's far point is free: a new node
    d["mesh"]["points"]["T"] = [L, 0.0, 3.0]
    d["mesh"]["lines"]["post"] = {"through": ["T", "A"], "size": 9.0}       # ...held by a post, built with the stay
    d["assign"].append({"region": "post", "section": "s", "orientation": [0, 1, 0]})
    d["steps"][1]["activate"] = ["stay", "post"]
    seen = {}

    def prog(count, *_):
        s = got["s"]
        if s.step_index == 1 and "u" not in seen:
            seen["u"] = s.dofs.view(s.q)[s._region("T")[0], :3].copy()
        return True
    out = run_case(d, on_ready=lambda s, _p: got.__setitem__("s", s), on_progress=prog, progress_every=1)
    assert out.status == "CONVERGED", out.message
    # T is reached through the post from A (clamped, unmoved), not through the stay from the tip B, which
    # has sagged 21 mm: after the step's first increment it has only what a tenth of the jack pulled it down
    assert -2.0e4 / KB < -0.020 and abs(seen["u"][2]) < 0.010
