"""The compiled relaxation (1.9): the iterations of a staged
relaxation of frames, bars and connectors under `backend: numba` run in one
compiled function (`kernels/numba_step.relax_chunk`), in chunks that end at
every progress check, where the loop reports and a checkpoint can be
written. Loop against compiled, to the bit: the state, the counters, the
ledger, the result series -- on a portal frame with a prescribed
displacement blended in over the ramp, with and without the mass-
proportional term, on one thread and on two, on a frame with a bar and a
connector, and through a stop and restart in the middle of a level.
"""
import os

import numpy as np
import pytest

from vfife.api import run_case

numba = pytest.importorskip("numba")


def _portal(tmp_path, name, *, damping=None, threads=1, mixed=False, prescribed=True, every=40,
            budget=4000):
    pts = {"A": [0, 0, 0], "B": [0, 0, 3.0], "C": [4.0, 0, 3.0], "D": [4.0, 0, 0]}
    lines = {"c1": {"through": ["A", "B"], "size": 0.5}, "g": {"through": ["B", "C"], "size": 0.5},
             "c2": {"through": ["D", "C"], "size": 0.5}}
    groups = {"frame": ["c1", "g", "c2"]}
    if mixed:
        # a bar from the girder's end to an anchor, and a bearing under it
        pts.update({"E": [8.0, 0, 3.0], "G": [4.0, 0, 3.5]})
        lines.update({"bar": {"through": ["C", "E"], "size": 4.0},
                      "brg": {"through": ["G", "C"], "size": 1.0}})
        groups["bars"] = ["bar"]; groups["brgs"] = ["brg"]
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts, "lines": lines, "groups": groups},
         "materials": {"s": {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}},
         "sections": {"b": {"type": "beam", "material": "s", "shape": "general", "area": 4e-3,
                            "iy": 3e-5, "iz": 2e-5, "j": 1e-5},
                      "t": {"type": "bar", "material": "s", "shape": "general", "area": 2e-4},
                      "k": {"type": "connector", "stiffness": [0, 1e8, 1e9, 1e6, 1e6, 1e6],
                            "laws": {"ux": {"type": "bilinear", "k1": 1e8, "k2": 1e6, "fy": 2e3}}}},
         "assign": [{"region": "frame", "section": "b", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "D", "dofs": [1, 1, 1, 0, 0, 0]}],
         "loads": [{"region": "B", "vector": [1.0e4, 0, -2.0e4]}],
         "solver": {"mode": "relaxation", "backend": "numba", "load_steps": 3, "tolerance": 1e-7,
                    "max_steps_per_load": budget, "n_steps": 3 * budget, "threads": threads},
         "output": {"path": str(tmp_path / f"{name}.h5"), "fields": ["displacement"], "every": 1,
                    "monitors": {"ux": {"kind": "displacement", "region": "C", "dof": "x"}}}}
    if prescribed:
        d["constraints"].append({"region": "C", "dofs": [0, 0, 1, 0, 0, 0], "value": -0.004})
    if damping is not None:
        d["solver"]["damping"] = damping
    if mixed:
        d["assign"] += [{"region": "bars", "section": "t"},
                        {"region": "brgs", "section": "k", "orientation": [0, 1, 0]}]
        d["constraints"] += [{"region": "E", "dofs": [1] * 6}, {"region": "G", "dofs": [1] * 6}]
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


STATE = ("q", "v", "orient", "end_actions", "step", "n_resets", "ke_prev", "_half", "_force_scale",
         "_motion_scale", "residuals", "kinetic")


def _same(s0, s1, o0, o1):
    for k in STATE:
        a, b = getattr(s0, k), getattr(s1, k)
        assert np.array_equal(np.asarray(a), np.asarray(b)), k
    l0, l1 = s0.diagnostics["energy"], s1.diagnostics["energy"]
    for k in ("external_work", "internal_work", "dissipated"):
        assert getattr(l0, k) == getattr(l1, k), k
    for k in o0.monitors:
        assert np.array_equal(o0.monitors[k], o1.monitors[k]), k


@pytest.mark.parametrize("damping,threads", [(None, 1), ({"type": "viscous", "coeff": 2e-4}, 1), (None, 2)])
def test_a_staged_frame_relaxation_compiled_is_the_loop_to_the_bit(tmp_path, damping, threads):
    d = _portal(tmp_path, f"p{threads}{int(damping is not None)}", damping=damping, threads=threads)
    o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "a0.h5"))), False)
    o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "a1.h5"))), True)
    assert o0.status == o1.status == "CONVERGED", (o0.message, o1.message)
    assert s0.compiled_relax_refusal == "VFIFE_COMPILED_STEP is off"
    assert s1.compiled_relax_refusal is None
    assert s1.n_resets > 3 and s1.step > 100
    _same(s0, s1, o0, o1)


def test_bars_and_a_yielding_bearing_compiled_are_the_loop_to_the_bit(tmp_path):
    d = _portal(tmp_path, "mx", mixed=True)
    o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "m0.h5"))), False)
    o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "m1.h5"))), True)
    assert o0.status == o1.status == "CONVERGED", (o0.message, o1.message)
    assert type(s1).__name__ == "MixedStructuralSolver" and s1.compiled_relax_refusal is None
    assert len(s1._connector_ids) == 1 and len(s1._truss_ids) == 1
    _same(s0, s1, o0, o1)
    for k in ("connector_deformation", "_conn_committed", "_conn_slip"):
        assert np.array_equal(getattr(s0, k), getattr(s1, k)), k
    assert np.abs(s1._conn_committed).max() > 0.0                # the bearing yielded


def test_a_relaxation_stopped_in_a_level_resumes_to_the_same_bits(tmp_path):
    """A stop at a progress check (where the compiled chunk ends) writes the
    checkpoint the loop writes, and the restart -- compiled or not -- ends
    where the uninterrupted run ends."""
    from vfife.core.diagnostics import AnalysisInterrupted
    d = _portal(tmp_path, "rs")
    _w, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "whole.h5"))), True, progress_every=70)
    ends = {}
    for compiled in (False, True):
        path = str(tmp_path / f"part_{compiled}.h5")
        calls = {"n": 0}

        def stop(*_a):
            calls["n"] += 1
            return calls["n"] < 4            # in the first level, at its fourth check

        with pytest.raises(AnalysisInterrupted):
            _run(dict(d, output=dict(d["output"], path=path)), compiled, on_progress=stop, progress_every=70)
        rest, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"rest_{compiled}.h5"))),
                        compiled, restart=path, progress_every=70)
        assert rest.status == "CONVERGED", rest.message
        ends[compiled] = sr
    for k in ("q", "v", "orient", "step", "n_resets", "ke_prev", "_force_scale", "_motion_scale"):
        assert np.array_equal(np.asarray(getattr(ends[False], k)), np.asarray(getattr(ends[True], k))), k
        assert np.array_equal(np.asarray(getattr(ends[True], k)), np.asarray(getattr(sw, k))), k


def test_the_implicit_fallback_relaxes_compiled(tmp_path):
    """`static.fallback: relaxation` runs `_relax_stage`: the compiled one."""
    d = _portal(tmp_path, "fb")
    d["solver"] = {"mode": "static", "backend": "numba", "tolerance": 1e-7, "load_steps": 2,
                   "static": {"fallback": "relaxation", "max_iterations": 1, "cutbacks": 0, "stabilise": False}}
    o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "f0.h5"))), False)
    o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "f1.h5"))), True)
    assert o0.status == o1.status, (o0.message, o1.message)
    assert getattr(s1, "compiled_relax_refusal", "never asked") is None
    for k in ("q", "orient", "step"):
        assert np.array_equal(np.asarray(getattr(s0, k)), np.asarray(getattr(s1, k))), k


def _floor(d):
    d["regions"] = {"floor": {"plane": {"point": [0, 0, 3.0], "normal": [0, 0, 1]}}}
    d["diaphragms"] = [{"region": "floor", "normal": "z"}]
    d["constraints"] = d["constraints"][:2]


@pytest.mark.parametrize("change,why", [
    (lambda d: d["solver"].update(backend="numpy"), "backend is not numba"),
    (_floor, "rigid diaphragms"),
    (lambda d: d["constraints"].__setitem__(2, {"region": "C", "dofs": [0, 0, 0, 0, 1, 0], "value": 1e-3}),
     "a prescribed rotation"),
])
def test_what_the_compiled_relaxation_refuses_it_says(tmp_path, change, why):
    d = _portal(tmp_path, "rf", budget=300)
    change(d)
    _o, s = _run(d, True)
    assert s.compiled_relax_refusal == why


@pytest.mark.parametrize("compiled", [False, True])
def test_a_stop_at_the_check_of_the_converging_iteration_resumes_exactly(tmp_path, compiled):
    """The progress check of the iteration that converges a level comes
    before the convergence test: a checkpoint written there carried the
    iteration BEFORE's measures, and the restart ran on past the level's
    end (found in the review of 1.9). Now it carries its
    own, and the restarted level ends at once."""
    from vfife.core.diagnostics import AnalysisInterrupted
    d = _portal(tmp_path, f"cv{int(compiled)}")
    calls = []
    _w, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "w.h5"))), compiled,
                  on_progress=lambda *a: calls.append(a[0]) and None, progress_every=1)
    n1 = calls.count(1) - 1                  # level 1's iterations (the last call is the level's own)
    k = {"n": 0}

    def stop(*_a):
        k["n"] += 1
        return k["n"] < n1                   # stop at the converging iteration's check

    path = str(tmp_path / "part.h5")
    with pytest.raises(AnalysisInterrupted):
        _run(dict(d, output=dict(d["output"], path=path)), compiled, on_progress=stop, progress_every=1)
    _r, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "r.h5"))), compiled,
                  restart=path, progress_every=1)
    for key in ("q", "v", "step", "n_resets", "ke_prev"):
        assert np.array_equal(np.asarray(getattr(sr, key)), np.asarray(getattr(sw, key))), key
