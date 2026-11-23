"""The contact between members in the compiled step (1.12).

A model with `contact.members` ran the step-by-step loop: the contact wrapped
the force evaluation (a component since v1.18), and the compiled step takes no step that is hooked.
One evaluation of the contact -- the neighbour list, the pairs that overlap,
their forces, the allowances -- is now ONE compiled function
(`kernels/numba_member_contact.contact_step`) that the loop's
`MemberContact.force` and the compiled step both call, so the two agree to
the bit by construction.

Each case runs with `VFIFE_COMPILED_STEP` off and on, and the state, the
allowances, the contact's forces, its counts, the ledger and the output file
are compared with `np.array_equal`:

* a beam landing on another and resting there, with friction, 1 and 2 threads;
* a chain of bars draped over a beam (points without rotary inertia);
* fibre beams (the fibre step's own chunk);
* a neighbour list that outgrows its arrays in the middle of a run: the
  kernel hands that step back and goes on;
* a bearing removed in the middle (the loop's step), a bar that breaks in
  contact, a restart;
* a fibre member that fractures while in contact (the chunk hands that step
  over half done);
* a hook over the contact's is still a hook;
* and a question asked before a step is not written after it.
"""
import os

import h5py
import numpy as np
import pytest

numba = pytest.importorskip("numba")

from test_member_contact import G, RHO, _girder_on_cap, _go, _two_beams      # noqa: E402
from vfife.core import compiled_step                                          # noqa: E402


FIBRE = {"type": "steel02", "E": 2.0e10, "Fy": 4.0e8, "b": 0.01, "R0": 20.0, "rho": RHO}


def _run(d, compiled, **kw):
    old = os.environ.get("VFIFE_COMPILED_STEP")
    os.environ["VFIFE_COMPILED_STEP"] = "1" if compiled else "0"
    try:
        return _go(d, **kw)
    finally:
        if old is None:
            os.environ.pop("VFIFE_COMPILED_STEP", None)
        else:
            os.environ["VFIFE_COMPILED_STEP"] = old


def _state(s):
    c = s.member_contact
    led = s.diagnostics["energy"]
    out = {"q": s.q, "v": s.v, "orient": s.orient, "keys": c.keys[c.vals > 0], "vals": c.vals[c.vals > 0],
           "f": c.nodal_force, "mom": c.nodal_moment, "pen": c.penetration,
           "counts": np.array([c.n_active, c.max_penetration, c.n_sweeps, s.step, float(c.stiffness_capped)]),
           "ledger": np.array([led.external_work, led.internal_work, led.dissipated], float)}
    return {k: np.array(v, copy=True) for k, v in out.items()}


def _file(path):
    out = {}
    with h5py.File(path) as f:
        def take(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiu":
                out[name] = np.asarray(obj)
        f.visititems(take)
    return {k: v for k, v in out.items() if "wall" not in k and "timing" not in k}


def _met(d):
    """The largest contact force in the frames of the case's output."""
    f = _file(d["output"]["path"])
    keys = [k for k in f if "contact_force" in k]
    assert keys, sorted(f)
    return max(float(np.abs(f[k]).max()) for k in keys)


def _same(a, b):
    assert sorted(a) == sorted(b)
    bad = [k for k in a if not np.array_equal(a[k], b[k], equal_nan=True)]
    assert not bad, bad


def _both(make, tmp_path, name, **kw):
    """(loop state, compiled state, compiled solver): the case of `make`
    run both ways, compared to the bit, files included."""
    res = []
    for compiled in (False, True):
        d = make(f"{name}{int(compiled)}")
        _o, s = _run(d, compiled, **kw)
        assert (s.compiled_step_refusal is None) == compiled, s.compiled_step_refusal
        res.append((_state(s), _file(d["output"]["path"]), s))
    _same(res[0][0], res[1][0])
    _same(res[0][1], res[1][1])
    return res[0][0], res[1][0], res[1][2]


@pytest.mark.parametrize("threads", [1, 2])
def test_a_beam_resting_on_another_with_friction(tmp_path, threads):
    def make(name):
        d = _two_beams(tmp_path, name, h=0.3, total=0.8, damping=1.0, friction=0.4,
                       struct_damping={"type": "viscous", "coeff": 6.0})
        d["loads"][0]["gravity"] = [0.2 * G, 0, -G]                  # it slides too
        d["solver"]["threads"] = threads
        return d
    _a, b, s = _both(make, tmp_path, f"rest{threads}")
    assert s.member_contact.n_active >= 1 and b["counts"][1] > 0.0          # in contact at the end
    assert np.abs(b["f"]).max() > 100.0 and b["counts"][2] >= 2             # and the list was made again


def test_a_chain_of_bars_over_a_beam(tmp_path):
    def make(name):
        pts = {"A": [-1.0, 0, 0], "B": [1.0, 0, 0], "C": [0, -1.0, 0.16], "D": [0, 1.0, 0.16]}
        return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
                "mesh": {"type": "lines", "points": pts,
                         "lines": {"beam": {"through": ["A", "B"], "size": 0.5},
                                   "chain": {"through": ["C", "D"], "size": 0.25}}},
                "materials": {"s": {"type": "linear_elastic", "E": 2.0e9, "nu": 0.3, "rho": RHO}},
                "sections": {"b": {"type": "beam", "material": "s", "shape": "rect", "width": 0.2, "height": 0.2},
                             "c": {"type": "bar", "material": "s", "shape": "circle", "diameter": 0.1}},
                "assign": [{"region": "beam", "section": "b", "orientation": [0, 0, 1]},
                           {"region": "chain", "section": "c"}],
                "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "B", "dofs": [1] * 6}],
                "loads": [{"region": "chain", "gravity": [0.5 * G, 0, -G], "amplitude": "hold"}],
                "contact": {"members": {"bodies": [{"region": "beam", "radius": 0.1}, {"region": "chain"}],
                                        "friction": 0.2, "damping": 1.0}},
                "solver": {"mode": "dynamic", "backend": "numba", "total_time": 0.3, "damping": {"type": "none"}},
                "output": {"path": str(tmp_path / f"{name}.h5"), "every": 50,
                           "fields": ["displacement", "contact_force", "contact_penetration"]}}
    _a, b, s = _both(make, tmp_path, "chain")
    assert np.abs(b["f"]).max() > 0.0 or b["q"].reshape(-1, 6)[:, 2].min() < -0.01
    assert np.abs(b["mom"][s.mesh.regions["chain"]]).max() == 0.0


def test_fibre_beams_in_contact(tmp_path):
    def make(name):
        d = _two_beams(tmp_path, name, h=0.3, total=0.25, damping=1.0, friction=0.3,
                       struct_damping={"type": "viscous", "coeff": 6.0})
        d["materials"]["f"] = FIBRE
        d["sections"]["b"] = {"type": "fiber", "material": "f", "shape": "rect", "width": 0.2, "height": 0.2,
                              "ny": 4, "nz": 4}
        return d
    _a, b, s = _both(make, tmp_path, "fib")
    assert type(s).__name__ == "FiberBeamSolver"
    # they met: the lower beam, held at its ends, was pushed down
    assert b["q"].reshape(-1, 6)[s.mesh.regions["lower"], 2].min() < -1e-4


def test_a_neighbour_list_that_outgrows_its_arrays_hands_the_step_back(tmp_path, monkeypatch):
    seen = []
    for kind in (compiled_step.Stepper, compiled_step.FiberStepper):
        orig = kind.advance

        def advance(self, first, last, _orig=orig):
            out = _orig(self, first, last)
            seen.append(out[0])
            return out
        monkeypatch.setattr(kind, "advance", advance)

    def short(s):
        c = s.member_contact
        c._ci = np.zeros(1, np.int64); c._cj = np.zeros(1, np.int64)

    for fibre in (False, True):
        def make(name):
            d = _two_beams(tmp_path, name, h=0.3, total=0.2, damping=1.0, friction=0.3)
            if fibre:
                d["materials"]["f"] = FIBRE
                d["sections"]["b"] = {"type": "fiber", "material": "f", "shape": "rect", "width": 0.2,
                                      "height": 0.2, "ny": 2, "nz": 2}
            return d
        ends = []
        for compiled in (False, True):
            seen.clear()
            d = make(f"cap{int(fibre)}{int(compiled)}")
            box = {}

            def ready(s, p):
                box["s"] = s
                short(s)
            old = os.environ.get("VFIFE_COMPILED_STEP")
            os.environ["VFIFE_COMPILED_STEP"] = "1" if compiled else "0"
            try:
                from vfife.api import run_case
                run_case(d, on_ready=ready)
            finally:
                os.environ.pop("VFIFE_COMPILED_STEP", None) if old is None else os.environ.update(
                    VFIFE_COMPILED_STEP=old)
            ends.append(_state(box["s"]))
            if compiled:
                # the first chunk was handed back at its first step, the rest ran compiled
                assert seen[0] in ("EVENT", "HANDBACK") and seen.count(0) >= 3, seen
                assert len(box["s"].member_contact._ci) > 1
        _same(*ends)


def test_a_bearing_removed_and_a_restart(tmp_path):
    from vfife.core.diagnostics import AnalysisInterrupted

    def make(name):
        d = _girder_on_cap(tmp_path, name, contact=True, removal=0.2)
        d["solver"]["total_time"] = 0.4
        d["contact"]["members"]["friction"] = 0.3
        d["output"]["fields"] = ["displacement", "contact_force"]
        return d
    a, b, s = _both(make, tmp_path, "rm")
    assert _met(make("rm1")) > 1.0e4                       # the girder landed on the cap
    # stopped in the compiled run and resumed: the same bits again
    part = make("part")
    n = {"k": 0}

    def stop(*_a):
        n["k"] += 1
        return n["k"] < 6
    with pytest.raises(AnalysisInterrupted):
        _run(part, True, on_progress=stop, progress_every=max(1, int(s.n_steps * 0.13)))
    rest = make("rest")
    _r, sr = _run(rest, True, restart=part["output"]["path"])
    assert sr.compiled_step_refusal is None
    c = _state(sr)
    for k in ("q", "v", "orient", "keys", "vals", "f", "mom", "pen"):
        assert np.array_equal(c[k], b[k]), k


def test_a_bar_that_breaks_in_contact(tmp_path):
    def make(name):
        pts = {"A": [-1.0, 0, 0], "B": [1.0, 0, 0], "C": [0, -1.0, 0.3], "D": [0, 1.0, 0.3]}
        return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
                "mesh": {"type": "lines", "points": pts,
                         "lines": {"beam": {"through": ["A", "B"], "size": 0.5},
                                   "chain": {"through": ["C", "D"], "size": 0.25}}},
                "materials": {"s": {"type": "linear_elastic", "E": 2.0e9, "nu": 0.3, "rho": RHO}},
                "sections": {"b": {"type": "beam", "material": "s", "shape": "rect", "width": 0.2, "height": 0.2},
                             "c": {"type": "bar", "material": "s", "shape": "circle", "diameter": 0.1}},
                "assign": [{"region": "beam", "section": "b", "orientation": [0, 0, 1]},
                           {"region": "chain", "section": "c", "failure": {"tensile_strength": 6.0e4}}],
                "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "B", "dofs": [1] * 6}],
                # the chain is pulled down at its two ends, over the beam
                "loads": [{"region": "chain", "gravity": [0, 0, -G], "amplitude": "hold"},
                          {"region": "C", "vector": [0, 0, -2.0e3], "amplitude": "hold"},
                          {"region": "D", "vector": [0, 0, -2.0e3], "amplitude": "hold"}],
                "contact": {"members": {"bodies": [{"region": "beam", "radius": 0.1}, {"region": "chain"}],
                                        "friction": 0.2, "damping": 1.0}},
                "solver": {"mode": "dynamic", "backend": "numba", "total_time": 0.3,
                           "damping": {"type": "viscous", "coeff": 2.0}},
                "output": {"path": str(tmp_path / f"{name}.h5"), "every": 50,
                           "fields": ["displacement", "contact_force"]}}
    _a, b, s = _both(make, tmp_path, "brk")
    events = s.failure_events
    assert len(events) >= 1, "no bar broke"
    assert np.abs(b["q"].reshape(-1, 6)[s.mesh.regions["chain"], 2]).max() > 0.05


def test_a_hook_over_the_contact_is_still_a_hook(tmp_path):
    d = _two_beams(tmp_path, "hook", total=0.01)
    _o, s = _run(d, True)
    assert s.compiled_step_refusal is None
    fixed = np.zeros(s.ndof, bool)
    why = lambda: compiled_step.refusal(s, fixed, fixed, [], [], [])           # noqa: E731
    assert why() is None
    # the contact is a component now, not a wrapper: the
    # solver's own methods are its class's
    assert "internal_force" not in vars(s) and "commit_state" not in vars(s)
    s.internal_force = lambda _f=s.internal_force: _f()
    assert "internal_force" in why()
    del s.internal_force
    assert why() is None
    s.commit_state = lambda *a, _c=s.commit_state: _c(*a)
    assert "commit_state" in why()
    del s.commit_state
    assert why() is None
    # ... and so is one on the contact itself
    c = s.member_contact
    c.force = lambda *a, _f=c.force, **k: _f(*a, **k)
    assert "the contact's force is hooked" in why()
    del c.force
    assert why() is None
    # ... or on its component
    comp = s.components[0]
    comp.line_force = lambda solver, out, _f=comp.line_force: _f(solver, out)
    assert "the contact's component is hooked" in why()
    del comp.line_force
    assert why() is None


@pytest.mark.parametrize("split,rider", [(False, 0.25), (True, 0.25), (False, 0.001)])
def test_a_fibre_member_that_fractures_in_contact(tmp_path, split, rider, monkeypatch):
    """The loop latches a fractured member -- it leaves the active set --
    BEFORE the contact of that step. The chunk learned of the fracture after
    its contact: the member stayed in one step longer, 5.5 mm apart at the
    end of this run (review of v1.12; the combination kept the loop
    in 1.12.0). The chunk now hands the step over half done, with the member
    forces it made.

    The thin rider makes the neighbour list sweep often (its skin is half
    the smallest radius), so that some of the handed-over steps are steps in
    which the chunk had already made the list: that sweep is the loop's to
    make and to count (the review's mutation: without the give-back,
    `n_sweeps` came out 3 too many)."""
    from test_compiled_breaking import _fibre_girder
    seen = []
    orig = compiled_step.FiberStepper.advance

    def advance(self, first, last):
        out = orig(self, first, last)
        seen.append(out[0])
        return out
    monkeypatch.setattr(compiled_step.FiberStepper, "advance", advance)

    def make(name):
        d = _fibre_girder(tmp_path, name, total=0.16)
        d["sections"]["girder"]["fracture"]["split"] = split
        # a second fibre beam held across the girder, 0.03 m under it
        d["mesh"]["points"].update({"R1": [4.5, -1.0, -0.53], "R2": [4.5, 1.0, -0.53]})
        d["mesh"]["lines"]["rider"] = {"through": ["R1", "R2"], "size": 0.5}
        d["assign"].append({"region": "rider", "section": "girder", "orientation": [0, 0, 1]})
        d["loads"][0] = {"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}
        d["constraints"] += [{"region": "R1", "dofs": [1] * 6}, {"region": "R2", "dofs": [1] * 6}]
        d["loads"][1]["vector"] = [0, 0, -2.0e6]
        d["contact"] = {"members": {"bodies": [{"region": "g", "radius": 0.25}, {"region": "rider", "radius": rider}],
                                    "damping": 1.0, "friction": 0.3}}
        d["output"]["fields"] = ["displacement", "fractured", "contact_force"]
        d["output"]["every"] = 20
        return d
    tag = f"ff{int(split)}{int(rider < 0.1)}"
    _a, b, s = _both(make, tmp_path, tag)
    assert len(np.flatnonzero(s.fracture.latch.fractured)) >= 2          # members fractured,
    if rider > 0.1:
        assert _met(make(tag + "1")) > 1.0e3                               # the two beams met,
    else:
        assert b["counts"][2] > 1000                                       # the list was made again and again,
    assert "HANDBACK" in seen and seen.count(0) > 10 * seen.count("HANDBACK")   # and the chunk ran the rest


def test_no_member_left_sweeps_once_a_step_on_both_paths(tmp_path):
    def make(name):
        d = _two_beams(tmp_path, name, total=0.05)
        d["materials"]["f"] = FIBRE
        d["sections"]["b"] = {"type": "fiber", "material": "f", "shape": "rect", "width": 0.2, "height": 0.2,
                              "ny": 2, "nz": 2}
        d["removals"] = [{"region": "upper", "time": 0.02}, {"region": "lower", "time": 0.02}]
        return d
    _both(make, tmp_path, "none")                         # the counts, n_sweeps among them


def test_a_question_before_a_step_is_not_written_after_it(tmp_path):
    """A probe leaves the allowances it arrived at aside, for a relaxation's
    level to commit. A dynamic step that followed it wrote its own -- and
    then, committing, wrote the probe's over them (1.11.0; found while the
    contact went into the compiled step)."""
    d = _girder_on_cap(tmp_path, "q", contact=True)
    d["solver"]["total_time"] = 0.02
    _o, s = _run(d, False)
    c = s.member_contact
    assert (c.vals > 0).any()
    q = s.q.reshape(-1, 6); v = s.v.reshape(-1, 6)
    x = s.mesh.nodes + q[:, :3]
    mt, rot = c.masses()
    lifted = x.copy(); lifted[s.mesh.regions["girder"], 2] += 0.3
    before = c.vals.copy()
    c.force(lifted, v[:, :3], v[:, 3:6], mt, rot, s.dt, True, commit=False)       # the question
    assert np.array_equal(c.vals, before) and c._pending is not None
    c.force(x, v[:, :3], v[:, 3:6], mt, rot, s.dt, True, commit=True)             # the step
    c.commit()                                                                   # ... and its commit_state
    assert np.array_equal(c.vals, before)                                        # at rest: nothing shrank


def test_a_second_contact_between_members_keeps_the_loop(tmp_path):
    """The compiled step runs the solver's own contact (`member_contact`)
    and nothing else: a second one a script attached ran in the loop only
    and the two paths differed by 0.2 m (review of v1.18)."""
    from vfife.core.member_contact import MemberContact
    d = _two_beams(tmp_path, "twice", total=0.01)
    _o, s = _run(d, True)
    fixed = np.zeros(s.ndof, bool)
    assert compiled_step.refusal(s, fixed, fixed, [], [], []) is None
    MemberContact(d["contact"]["members"], s).attach()
    assert "more than one contact between members" in compiled_step.refusal(s, fixed, fixed, [], [], [])
