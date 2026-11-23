"""Prescribed motions in the compiled step (1.6).

The compiled step left every step with a prescribed
motion to the step-by-step loop. It now takes what `orientation._apply_bc`
knows on translations -- a displacement with its amplitude (and its rate
when the amplitude is a record), a velocity, an acceleration, and a
support's own displacement, velocity or acceleration record, the
multiple-support form of a ground motion -- and books the ledger as the
loop does at a driven support (the structure receives fint + damp through
the reaction). Each case runs with `VFIFE_COMPILED_STEP` off and on and the
state, the kinematic velocities, the ledger and the monitors are compared
with `np.array_equal`:

* a portal pushed at one column top by a displacement table, pulled at the
  other by a velocity, and shaken at its beam by an acceleration, as a
  frame-and-bar model and as a fibre model;
* a girder on three piers under a displacement record travelling as a
  wave (case_36's form), under a velocity record and under an acceleration
  record, with Rayleigh damping;
* threads: the threaded step is the serial one to the bit;
* a restart from a stopped run under prescribed motions;
* a prescribed rotation is still refused.
"""
import os

import numpy as np
import pytest

from vfife.api import run_case
from vfife.core.diagnostics import AnalysisInterrupted

numba = pytest.importorskip("numba")


def _portal(tmp_path, name, fibre=False, total=0.02, threads=1, every=50, rot=False):
    H, B = 4.0, 6.0
    if fibre:
        sec = {"type": "fiber", "material": "c", "shape": "rect", "width": 0.4, "height": 0.6,
               "ny": 2, "nz": 6, "integration": "lobatto3",
               "bars": [{"y": y, "z": z, "section": "bar"} for y in (-0.15, 0.15) for z in (-0.25, 0.25)]}
        mats = {"c": {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.0035,
                      "ft": 2.5e6, "Ets": 1.5e9, "rat": 0.1, "rho": 2500},
                "s": {"type": "steel02", "E": 2.0e11, "Fy": 4.0e8, "b": 0.01, "R0": 20.0, "rho": 7850}}
        extra = {"bar": {"type": "bar", "material": "s", "shape": "circle", "diameter": 0.025}}
    else:
        sec = {"type": "beam", "material": "c", "shape": "rect", "width": 0.4, "height": 0.6}
        mats = {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2500},
                "s": {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850}}
        extra = {"tie": {"type": "bar", "material": "s", "shape": "general", "area": 5e-4}}
    lines = {"c1": {"through": ["A", "C"], "size": 1.0}, "c2": {"through": ["B", "D"], "size": 1.0},
             "g": {"through": ["C", "M", "D"], "size": 1.0}}
    assign = [{"region": "c1", "section": "girder", "orientation": [0, 1, 0]},
              {"region": "c2", "section": "girder", "orientation": [0, 1, 0]},
              {"region": "g", "section": "girder", "orientation": [0, 1, 0]}]
    if not fibre:
        lines["t"] = {"through": ["A", "D"], "size": 10.0}
        assign.append({"region": "t", "section": "tie"})
    push = {"region": "C", "dofs": [1, 0, 0, 0, 0, 0], "value": [0.02, 0, 0], "amplitude": "push"}
    if rot:
        push = {"region": "C", "dofs": [0, 0, 0, 0, 1, 0], "value": [0, 0, 0, 0, 0.002, 0], "amplitude": "push"}
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines",
                  "points": {"A": [0, 0, 0], "B": [B, 0, 0], "C": [0, 0, H], "D": [B, 0, H], "M": [B / 2, 0, H]},
                  "lines": lines},
         "materials": mats,
         "sections": dict(extra, girder=sec),
         "assign": assign,
         "constraints": [{"region": "A", "dofs": [1] * 6}, {"region": "B", "dofs": [1] * 6},
                         push,
                         {"region": "D", "dofs": [1, 0, 0, 0, 0, 0], "value": [0.5, 0, 0], "motion": "velocity",
                          "amplitude": "pull"},
                         {"region": "M", "dofs": [0, 0, 1, 0, 0, 0], "value": [0, 0, 30.0], "motion": "acceleration",
                          "amplitude": "shake"}],
         "amplitudes": {"push": [[0, 0], [0.5, 1.0], [1.0, 0.3]], "pull": [[0, 1.0], [1.0, -1.0]],
                        "shake": [[0, 0], [0.25, 1.0], [0.5, -1.0], [0.75, 1.0], [1.0, 0.0]]},
         "loads": [{"region": "all", "gravity": [0, 0, -9.80665], "amplitude": "hold"}],
         "solver": {"mode": "dynamic", "backend": "numba", "threads": threads, "total_time": total,
                    "damping": {"type": "viscous", "coeff": 1.0}},
         "output": {"path": str(tmp_path / f"{name}.h5"), "every": every, "fields": ["displacement"],
                    "monitors": {"uD": {"kind": "displacement", "region": "D", "dof": "x"},
                                 "wM": {"kind": "displacement", "region": "M", "dof": "z"},
                                 "RA": {"kind": "reaction", "region": "A", "dof": "x"},
                                 "RC": {"kind": "reaction", "region": "C", "dof": "x"}}}}
    return d


def _girder_on_piers(tmp_path, name, quantity, total=0.02, threads=1, wave=True):
    """Three piers with the deck between them, the piers' feet following a
    record: a sine as case_36's, written here."""
    L, H = 20.0, 8.0
    rec = tmp_path / f"{name}_rec.txt"
    t = np.linspace(0.0, 0.5, 1001)
    if quantity == "displacement":
        y = 0.05 * np.sin(2 * np.pi * 5 * t)
    elif quantity == "velocity":
        y = 0.05 * 2 * np.pi * 5 * np.cos(2 * np.pi * 5 * t)
    else:
        y = -0.05 * (2 * np.pi * 5) ** 2 * np.sin(2 * np.pi * 5 * t)
    np.savetxt(rec, np.column_stack([t, y]))
    record = {"path": str(rec), "quantity": quantity}
    if wave:
        record["wave"] = {"speed": 400.0, "direction": [1, 0, 0], "origin": [0, 0, 0]}
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines",
                  "points": {"F1": [0, 0, 0], "F2": [L, 0, 0], "F3": [2 * L, 0, 0],
                             "T1": [0, 0, H], "T2": [L, 0, H], "T3": [2 * L, 0, H]},
                  "lines": {"p1": {"through": ["F1", "T1"], "size": 2.0}, "p2": {"through": ["F2", "T2"], "size": 2.0},
                            "p3": {"through": ["F3", "T3"], "size": 2.0},
                            "deck": {"through": ["T1", "T2", "T3"], "size": 2.0}}},
         "materials": {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2500}},
         "sections": {"pier": {"type": "beam", "material": "c", "shape": "circle", "diameter": 1.5},
                      "deck": {"type": "beam", "material": "c", "shape": "rect", "width": 4.0, "height": 1.5}},
         "assign": [{"region": "p1", "section": "pier", "orientation": [1, 0, 0]},
                    {"region": "p2", "section": "pier", "orientation": [1, 0, 0]},
                    {"region": "p3", "section": "pier", "orientation": [1, 0, 0]},
                    {"region": "deck", "section": "deck", "orientation": [0, 1, 0]}],
         "regions": {"feet": {"box": [-1, 2 * L + 1, -1, 1, -0.1, 0.1]}},
         "constraints": [{"region": "feet", "dofs": [1, 1, 1, 1, 1, 1]},
                         {"region": "feet", "dofs": [1, 0, 0, 0, 0, 0], "record": record}],
         "loads": [{"region": "all", "gravity": [0, 0, -9.80665], "amplitude": "hold"}],
         "solver": {"mode": "dynamic", "backend": "numba", "threads": threads, "total_time": total,
                    "damping": {"type": "rayleigh", "alpha": 0.4, "beta": 1e-4}},
         "output": {"path": str(tmp_path / f"{name}.h5"), "every": 40, "fields": ["displacement"],
                    "monitors": {"uT2": {"kind": "displacement", "region": "T2", "dof": "x"},
                                 "uF3": {"kind": "displacement", "region": "F3", "dof": "x"},
                                 "R1": {"kind": "reaction", "region": "F1", "dof": "x"}}}}
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


def _same_checkpoint(a, b):
    ca, cb = a.checkpoint(), b.checkpoint()
    assert set(ca) == set(cb)
    for k in ca:
        if np.asarray(ca[k]).dtype.kind in "SUO":
            continue
        assert np.array_equal(np.asarray(ca[k]), np.asarray(cb[k])), k


def _same(d, tmp_path, name):
    o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}0.h5"))), False)
    o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}1.h5"))), True)
    assert o0.status == o1.status == "COMPLETED_DYNAMIC", (o0.message, o1.message)
    assert s1.compiled_step_refusal is None, s1.compiled_step_refusal
    for k in ("q", "v", "orient", "_kinematic_v"):
        assert np.array_equal(getattr(s0, k), getattr(s1, k)), k
    assert np.array_equal(np.asarray(s0.end_actions), np.asarray(s1.end_actions))
    l0, l1 = s0.diagnostics["energy"], s1.diagnostics["energy"]
    assert (l0.external_work, l0.internal_work, l0.dissipated) == (l1.external_work, l1.internal_work, l1.dissipated)
    assert np.abs(l0.external_work) > 0.0
    for k in o0.monitors:
        assert np.array_equal(o0.monitors[k], o1.monitors[k]), k
    _same_checkpoint(s0, s1)
    return s0, s1


def test_a_portal_pushed_pulled_and_shaken_as_a_frame(tmp_path):
    s0, s1 = _same(_portal(tmp_path, "pf"), tmp_path, "pf")
    assert type(s1).__name__ == "MixedStructuralSolver"
    qv = s1.dofs.view(s1.q)
    assert qv[:, 0].max() > 0.005                        # the push arrived
    assert np.abs(s1._kinematic_v).max() > 0.0           # the velocity and the acceleration moved theirs


def test_a_portal_pushed_pulled_and_shaken_as_a_fibre_model(tmp_path):
    s0, s1 = _same(_portal(tmp_path, "pb", fibre=True), tmp_path, "pb")
    assert type(s1).__name__ == "FiberBeamSolver" and s1._kernel is not None
    assert np.array_equal(s0._state_c, s1._state_c) and np.array_equal(s0._state_s, s1._state_s)


@pytest.mark.parametrize("quantity", ["displacement", "velocity", "acceleration"])
def test_piers_following_their_own_records(tmp_path, quantity):
    s0, s1 = _same(_girder_on_piers(tmp_path, f"gp_{quantity}", quantity), tmp_path, f"gp_{quantity}")
    assert np.any(s1._driven)
    assert np.abs(s1.dofs.view(s1.q)[:, 0]).max() > 1e-4  # the feet moved
    assert s0._ledger_kinetic() == s1._ledger_kinetic()


def test_threads_are_the_serial_answer_under_prescribed_motions(tmp_path):
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    _o1, s1 = _run(_girder_on_piers(tmp_path, "t1", "displacement"), True)
    _o2, s2 = _run(_girder_on_piers(tmp_path, "t2", "displacement", threads=2), True)
    assert s2.compiled_step_refusal is None
    _same_checkpoint(s1, s2)
    _o3, s3 = _run(_portal(tmp_path, "t3", fibre=True), True)
    _o4, s4 = _run(_portal(tmp_path, "t4", fibre=True, threads=2), True)
    _same_checkpoint(s3, s4)


def test_a_restart_under_prescribed_motions_resumes_where_the_run_stopped(tmp_path):
    d = _girder_on_piers(tmp_path, "rs", "acceleration", total=0.02)
    _w, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "w.h5"))), True, progress_every=100)
    ends = {}
    for compiled in (False, True):
        path = str(tmp_path / f"p{compiled}.h5")
        calls = {"n": 0}

        def stop(*_a):
            calls["n"] += 1
            return calls["n"] < 2

        with pytest.raises(AnalysisInterrupted):
            _run(dict(d, output=dict(d["output"], path=path)), compiled, on_progress=stop, progress_every=100)
        rest, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"r{compiled}.h5"))), compiled,
                        restart=path, progress_every=100)
        assert rest.status == "COMPLETED_DYNAMIC"
        ends[compiled] = sr
    _same_checkpoint(ends[False], ends[True])
    _same_checkpoint(ends[True], sw)


def test_a_prescribed_rotation_is_still_refused(tmp_path):
    _o, s = _run(_portal(tmp_path, "rr", total=0.002, rot=True), True)
    assert "prescribed rotation" in (s.compiled_step_refusal or "")
