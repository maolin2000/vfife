"""1.3, item 10: the explicit dynamic step as one compiled call.

`core/compiled_step` runs the runs of steps between two progress or output
points inside one compiled function. It is only worth having if it is the
step-by-step loop's answer TO THE BIT, so every check here runs the same
case twice -- `VFIFE_COMPILED_STEP=0` and on -- and compares the state the
two leave (positions, velocities, orientations), the energy ledger, the
monitors and the reactions with `np.array_equal`:

* a portal frame with two cable stays (frames and tension-only bars in a
  mixed mesh), under gravity, a tabulated point load, a uniform ground
  motion, and no damping, Rayleigh (alpha M + beta K) or viscous damping;
* the same frame without the stays (the pure frame solver);
* eleven fixed-shape loads with their own curves (the stacked groups of
  `BeamSolver._stacked_groups`), on one shared set of points and on
  different ones;
* threads: the threaded compilation is the serial one to the bit;
* a restart from a stopped run resumes where it stopped, the half-step
  restart included;
* a run that blows up stops at the same step with the same message, and a
  member that collapses is refused as it was.

And the step is refused -- the loop takes it -- wherever there is code of its
own: a hinge, a break, a contact, a prescribed rotation (a prescribed
translation, a velocity, an acceleration and a support's record are taken
since 1.6: tests/test_compiled_prescribed.py).
"""
import os

import h5py
import numpy as np
import pytest

from vfife.api import run_case
from vfife.core.diagnostics import AnalysisInterrupted

numba = pytest.importorskip("numba")

H, B = 4.0, 6.0


def _case(tmp_path, name, *, stays=True, damping=None, ground=False, stacked=None,
          total=0.05, threads=1, extra_loads=(), every=100):
    pts = {"A": [0, 0, 0], "Bt": [0, 0, H], "C": [B, 0, H], "D": [B, 0, 0]}
    lines = {"col1": {"through": ["A", "Bt"], "size": 1.0},
             "beam": {"through": ["Bt", "C"], "size": 1.0},
             "col2": {"through": ["C", "D"], "size": 1.0}}
    # one section on `all` is the frame solver; several assignments a mixed mesh
    assign = ([{"region": r, "section": "g", "orientation": [0, 1, 0] if r != "beam" else [0, 0, 1]}
               for r in ("col1", "beam", "col2")] if stays else
              [{"region": "all", "section": "g", "orientation": [0, 1, 0]}])
    sections = {"g": {"type": "beam", "material": "s", "shape": "rect", "width": 0.3, "height": 0.4}}
    if stays:
        pts["E"] = [B / 2, 3.0, 2 * H]
        lines["st1"] = {"through": ["Bt", "E"], "size": 10.0}
        lines["st2"] = {"through": ["C", "E"], "size": 10.0}
        sections["c"] = {"type": "bar", "material": "s", "shape": "general", "area": 2e-4}
        assign += [{"region": "st1", "section": "c", "tension_only": True},
                   {"region": "st2", "section": "c", "tension_only": True}]
    loads = [{"region": "all", "gravity": [0, 0, -9.80665], "amplitude": "hold"},
             {"region": "C", "vector": [4.0e4, 1.0e4, 0.0], "amplitude": "push"}]
    amps = {"push": [[0, 0], [0.3, 1.0], [0.6, -0.5], [1.0, 0.2]]}
    if ground:
        rec = tmp_path / f"{name}_ag.txt"
        t = np.linspace(0.0, total, 41)
        np.savetxt(rec, np.column_stack([t, 3.0 * np.sin(2 * np.pi * 15 * t)]))
        loads.append({"ground_motion": {"path": str(rec), "direction": [0.8, 0.6, 0.0]}})
    if stacked is not None:
        for i in range(11):
            nm = f"w{i}"
            if stacked == "shared":
                amps[nm] = [[0, 0], [0.5, 1.0 + 0.1 * i], [1.0, -0.3 * i]]
            else:
                amps[nm] = [[0, 0], [0.2 + 0.05 * i, 1.0], [1.0, 0.1 * i]]
            loads.append({"region": ("Bt", "C", "E" if stays else "C")[i % 3],
                          "vector": [100.0 * (i + 1), -50.0 * i, 30.0], "amplitude": nm})
    loads += list(extra_loads)
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts, "lines": lines},
         "materials": {"s": {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}},
         "sections": sections, "assign": assign, "amplitudes": amps,
         "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "D", "dofs": [1, 1, 1, 0, 0, 0]}]
         + ([{"region": "E", "dofs": [1] * 6}] if stays else []),
         "loads": loads,
         "solver": {"mode": "dynamic", "backend": "numba", "threads": threads, "total_time": total},
         "output": {"path": str(tmp_path / f"{name}.h5"), "every": every,
                    "fields": ["displacement", "rotation"],
                    "monitors": {"ux": {"kind": "displacement", "region": "C", "dof": "x"},
                                 "RA": {"kind": "reaction", "region": "A", "dof": "x"}}}}
    if damping is not None:
        d["solver"]["damping"] = damping
    return d


def _run(d, compiled, **kw):
    old = os.environ.get("VFIFE_COMPILED_STEP")
    os.environ["VFIFE_COMPILED_STEP"] = "1" if compiled else "0"
    box = {}
    try:
        out = run_case(d, on_ready=lambda s, p: box.setdefault("s", s), **kw)
    finally:
        if old is None:
            os.environ.pop("VFIFE_COMPILED_STEP", None)
        else:
            os.environ["VFIFE_COMPILED_STEP"] = old
    return out, box["s"]


def _state(s):
    led = s.diagnostics["energy"]
    return {"q": s.q.copy(), "v": s.v.copy(), "orient": s.orient.copy(), "step": s.step,
            "ledger": np.array([led.external_work, led.internal_work, led.dissipated]),
            "end_actions": np.asarray(s.end_actions).copy()}


def _same_checkpoint(a, b):
    """Everything a restart would carry on from, entry by entry."""
    ca, cb = a.checkpoint(), b.checkpoint()
    assert set(ca) == set(cb)
    for k in ca:
        if np.asarray(ca[k]).dtype.kind in "SUO":
            continue
        assert np.array_equal(np.asarray(ca[k]), np.asarray(cb[k])), k


def _same(d, tmp_path, name):
    d0 = dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}_loop.h5")))
    d1 = dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}_comp.h5")))
    o0, s0 = _run(d0, False)
    o1, s1 = _run(d1, True)
    assert o0.status == o1.status == "COMPLETED_DYNAMIC", (o0.message, o1.message)
    assert s0.compiled_step_refusal == "VFIFE_COMPILED_STEP is off"
    assert s1.compiled_step_refusal is None, s1.compiled_step_refusal
    a, b = _state(s0), _state(s1)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    _same_checkpoint(s0, s1)
    for k in o0.monitors:
        assert np.array_equal(o0.monitors[k], o1.monitors[k]), k
    with h5py.File(d0["output"]["path"]) as f0, h5py.File(d1["output"]["path"]) as f1:
        for g in ("diagnostics",):
            for k in f0[g]:
                if isinstance(f0[g][k], h5py.Dataset):
                    assert np.array_equal(np.asarray(f0[g][k]), np.asarray(f1[g][k])), (g, k)
    return s0, s1


@pytest.mark.parametrize("damping", [None, {"type": "rayleigh", "alpha": 0.8, "beta": 2e-5},
                                     {"type": "viscous", "coeff": 1.5}])
def test_frame_and_stays_the_loops_answer_to_the_bit(tmp_path, damping):
    s0, s1 = _same(_case(tmp_path, "fs", damping=damping, ground=True), tmp_path, "fs")
    # the stays did go slack somewhere: the tension-only branch ran
    assert type(s1).__name__ == "MixedStructuralSolver"


def test_the_pure_frame_solver_too(tmp_path):
    _s0, s1 = _same(_case(tmp_path, "pf", stays=False, ground=True,
                          damping={"type": "rayleigh", "alpha": 0.3, "beta": 1e-5}), tmp_path, "pf")
    assert type(s1).__name__ == "BeamSolver"


@pytest.mark.parametrize("shape", ["shared", "own"])
def test_stacked_load_groups(tmp_path, shape):
    _same(_case(tmp_path, "sg", stacked=shape), tmp_path, "sg")


def test_threads_are_the_serial_answer(tmp_path):
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    d1 = _case(tmp_path, "t1", ground=True, damping={"type": "rayleigh", "alpha": 0.5, "beta": 1e-5})
    d2 = _case(tmp_path, "t2", ground=True, damping={"type": "rayleigh", "alpha": 0.5, "beta": 1e-5},
               threads=2)
    _o1, s1 = _run(d1, True)
    _o2, s2 = _run(d2, True)
    assert s2.compiled_step_refusal is None
    a, b = _state(s1), _state(s2)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_a_restart_resumes_where_the_run_stopped(tmp_path):
    """Stopped at a progress check, restarted from its checkpoint: the
    compiled step picks up at the next step and ends where one unbroken run
    ends -- and where the step-by-step loop's restart ends."""
    d = _case(tmp_path, "rs", ground=True, damping={"type": "viscous", "coeff": 1.0}, every=50)
    _whole, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "whole.h5"))), True,
                      progress_every=100)
    ends = {}
    for compiled in (False, True):
        path = str(tmp_path / f"part_{compiled}.h5")
        calls = {"n": 0}

        def stop(*_a):
            calls["n"] += 1
            return calls["n"] < 2           # stop at the second progress check

        with pytest.raises(AnalysisInterrupted):
            _run(dict(d, output=dict(d["output"], path=path)), compiled,
                 on_progress=stop, progress_every=100)
        rest, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"rest_{compiled}.h5"))),
                        compiled, restart=path, progress_every=100)
        assert rest.status == "COMPLETED_DYNAMIC"
        ends[compiled] = sr
    _same_checkpoint(ends[False], ends[True])
    _same_checkpoint(ends[True], sw)


@pytest.mark.parametrize("threads", [1, 2])
def test_a_run_that_blows_up_stops_at_the_same_step(tmp_path, threads):
    """Far above the stable step the frame turns inside out. Both the loop and
    the compiled step stop as NONFINITE at the same step (the kernels divide
    as IEEE does since this stage; they used to raise ZeroDivisionError, and
    from a parallel region a SystemError)."""
    if threads > 1 and numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    d = _case(tmp_path, "bu", stays=False, threads=threads)
    d["solver"]["dt_safety"] = 2.5
    d["solver"]["total_time"] = 0.2
    ends = []
    for compiled in (False, True):
        out, sv = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"bu{compiled}.h5"))), compiled)
        assert out.status == "NONFINITE", (out.status, out.message)
        ends.append((out.message, sv.step))
    assert ends[0] == ends[1]


def test_a_collapsed_member_is_refused_as_before(tmp_path):
    from vfife.core import compiled_step
    d = _case(tmp_path, "cm")
    _o, s = _run(d, True)
    fixed, moving, prescribed, groups, ground = s.step_data(0)
    st = compiled_step.stepper(s, fixed, moving, prescribed, groups, ground)
    assert st is not None
    e = int(np.flatnonzero(s.member_kind != "truss2")[3])
    n1, n2 = s.mesh.elements[e][:2]
    qv = s.dofs.view(s.q)
    qv[n2, :3] = qv[n1, :3] + s.mesh.nodes[n1] - s.mesh.nodes[n2]       # both ends at one point
    step = s.step
    with pytest.raises(ValueError, match="collapsed to zero length"):
        st.advance(1, 5)
    assert s.step == step                                                # nothing was taken


@pytest.mark.parametrize("why, change", [
    ("a prescribed rotation", lambda d: d["constraints"].append(
        {"region": "C", "dofs": [0, 0, 0, 0, 1, 0], "value": [0, 0, 0, 0, 1e-3, 0]})),
    ("hooked", lambda d: d.update(contact={"rigid": [{"name": "floor", "region": "all",
                                                     "plane": {"point": [0, 0, -1.0], "normal": [0, 0, 1]}}]})),
    ("rigid diaphragms", lambda d: d.update(diaphragms=[{"region": "beam", "normal": "z"}])),
])
def test_the_step_is_refused_where_it_has_code_of_its_own(tmp_path, why, change):
    d = _case(tmp_path, "rf", total=0.002)
    change(d)
    out, s = _run(d, True)
    assert s.compiled_step_refusal is not None
    assert why.split()[0] in s.compiled_step_refusal or why in s.compiled_step_refusal, s.compiled_step_refusal


def test_off_by_environment(tmp_path):
    _o, s = _run(_case(tmp_path, "off", total=0.002), False)
    assert s.compiled_step_refusal == "VFIFE_COMPILED_STEP is off"


def test_an_interrupt_as_the_call_returns_books_the_steps_taken(tmp_path, monkeypatch):
    """A KeyboardInterrupt delivered as the compiled call returns: the state
    has taken the whole run of steps, and the step count, the ledger and the
    restart factor say so (found in review: they stayed behind)."""
    from vfife.core import compiled_step
    from vfife.kernels import numba_step
    d = _case(tmp_path, "ki", total=0.01)
    _o, s = _run(d, True)
    fixed, moving, prescribed, groups, ground = s.step_data(0)
    st = compiled_step.stepper(s, fixed, moving, prescribed, groups, ground)
    real = numba_step.dynamic_chunk

    def interrupted(*a):
        real(*a)
        raise KeyboardInterrupt
    monkeypatch.setattr(numba_step, "dynamic_chunk", interrupted)
    s._half = True
    step0 = s.step
    with pytest.raises(KeyboardInterrupt):
        st.advance(1, 37)
    assert s.step == step0 + 37 and s._half is False
    assert s.diagnostics["energy"]._u is not None and np.array_equal(s.diagnostics["energy"]._u, s.q) is False


def test_a_method_replaced_on_the_class_is_refused(tmp_path, monkeypatch):
    from vfife.core.mixed_structural_solver import MixedStructuralSolver
    from vfife.core import compiled_step
    d = _case(tmp_path, "cls", total=0.002)
    _run(d, True)                                     # the module has seen the class as it is
    orig = MixedStructuralSolver.internal_force
    monkeypatch.setattr(MixedStructuralSolver, "internal_force", lambda self: orig(self))
    _o, s = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "cls2.h5"))), True)
    assert "internal_force" in s.compiled_step_refusal
    assert compiled_step.next_stop(1, 10 ** 6, 10 ** 6, 10 ** 6, cap=7) == 7
