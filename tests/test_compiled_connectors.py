"""The connectors compiled, and the compiled step on bearings and on fibre
beams.

`kernels/numba_connector` is `elements.connector.response` one
configuration at a time: the deformation after the rigid motion, its 6 x 12
finite-difference Jacobian over 25 configurations, the stiffness, the
curves, the laws (gap, bilinear, friction, the planar pair) and the dashpot.
It is the reference to round-off, not to the bit (NumPy's vectorised
transcendental functions round differently); the Jacobian's central
differences turn 1e-16 in a deformation into about 1e-9 of a force, which
is what the comparison allows.

Once the connectors are compiled, the step that carries them is too: the
frame solvers' take connectors after their bars, and the fibre
solver gets a step of its own around its fused kernel. Each runs the same
case with `VFIFE_COMPILED_STEP` off and on and compares the state, the
fibres' and the laws' history, the ledger and the monitors with
`np.array_equal`:

* a girder on two bearings -- one with a dashpot, one sliding on friction
  in the plane against the weight it carries, with a gap -- shaken along the
  bridge and pushed, as a frame (mixed line) and as a fibre beam;
* threads: the threaded fibre kernel inside the compiled step is the serial
  one to the bit;
* a restart from a stopped run;
* a short fibre run keeps the loop; a fibre model's bearings are held to
  the connector rules. (Fracture is taken since 1.6: tests/test_compiled_breaking.py.)
"""
import os

import numpy as np
import pytest

from vfife.api import run_case
from vfife.core.diagnostics import AnalysisInterrupted

numba = pytest.importorskip("numba")


# --------------------------------------------------------------- the kernel
def _props():
    return [
        {"stiffness": [5e10, 1e9, 1e10, 1e10, 1e10, 1e7]},
        {"stiffness": [5e10, 7e7, 1e10, 1e10, 1e10, 1e7], "coordinate_system": "global"},
        {"stiffness": [1e9] * 6, "connector_damping": [1e5] * 6,
         "curves": {"uy": [[-0.01, -2e6], [0, 0], [0.01, 1e6], [0.03, 1.5e6]]}},
        {"stiffness": [1e9, 1e6, 1e6, 1e8, 1e8, 1e8], "connector_damping": [1e4] * 6,
         "laws": {"ux": {"type": "gap", "stiffness": 2e9, "lo": 0.0},
                  "uy": {"type": "friction", "stiffness": 1e8, "mu": 0.05, "normal": "ux"},
                  "uz": {"type": "friction", "stiffness": 1e8, "mu": 0.05, "normal": "ux"}}},
        {"stiffness": [1e9] * 6,
         "laws": {"uy": {"type": "bilinear", "k1": 1e8, "k2": 1e6, "fy": 2e5},
                  "uz": {"type": "bilinear", "k1": 1e8, "k2": 1e6, "fy": 2e5},
                  "rz": {"type": "bilinear", "k1": 1e7, "k2": 0.0, "fy": 1e4},
                  "ux": {"type": "gap", "stiffness": 3e9, "lo": -0.001, "hi": 0.002}}},
        {"stiffness": [1e9] * 6, "laws": {"uy": {"type": "friction", "stiffness": 1e8, "mu": 0.1,
                                                 "normal": 1e6}}},
    ]


def test_the_compiled_connector_is_the_reference_to_round_off():
    from vfife.core.rotations import exp_map
    from vfife.elements import connector as ref
    from vfife.kernels import numba_connector as nc
    props = _props()
    K, C, local, has_c, ncur, cx, cy, law, lp, planar = nc.encode(props)
    sp = nc.spins()
    rng = np.random.default_rng(3)
    for trial in range(300):
        c = trial % len(props)
        x0 = rng.standard_normal((2, 3))
        R0 = exp_map(rng.standard_normal(3)[None])[0]
        scale = 10 ** rng.uniform(-4, -1)
        u = rng.standard_normal((2, 3)) * scale
        T = exp_map(rng.standard_normal((2, 3)) * (0.3 if trial % 3 else 1e-8))
        ve = rng.standard_normal(12)
        sc = rng.standard_normal(6) * scale * 0.3
        so_r = np.zeros(6)
        # every other trial with the deformation the connector was born with (1.13)
        d0 = rng.standard_normal(6) * scale * 0.2 if trial % 2 else np.zeros(6)
        fe_r, act_r, d_r = ref.response(u, T, ve, x0, R0, props[c], include_damping=True,
                                        slip_committed=sc, slip_out=so_r, offset=d0 if trial % 2 else None)
        fe = np.zeros(12); act = np.zeros(6); d = np.zeros(6); so = np.zeros(6)
        nc.response_one(u, T, ve, x0, R0, sp, K[c], C[c], local[c], has_c[c], ncur[c], cx[c], cy[c],
                        law[c], lp[c], planar[c], sc, fe, act, d, so, True, d0)
        assert np.abs(d - d_r).max() <= 1e-12 * max(np.abs(d_r).max(), 1e-30), trial
        assert np.abs(fe - fe_r).max() <= 1e-7 * np.abs(fe_r).max(), trial
        assert np.abs(act - act_r).max() <= 1e-7 * max(np.abs(act_r).max(), 1e-30), trial
        if np.abs(so_r).max() > 0:
            assert np.abs(so - so_r).max() <= 1e-7 * np.abs(so_r).max(), trial


# ----------------------------------------------------- a girder on bearings
def _girder(tmp_path, name, fibre=False, total=0.03, threads=1, every=50):
    L = 10.0
    rec = tmp_path / f"{name}_ag.txt"
    t = np.linspace(0.0, max(total, 0.05), 81)
    np.savetxt(rec, np.column_stack([t, 4.0 * np.sin(2 * np.pi * 8 * t)]))
    if fibre:
        girder = {"type": "fiber", "material": "c", "shape": "rect", "width": 0.6, "height": 0.8,
                  "ny": 4, "nz": 6, "integration": "lobatto3",
                  "bars": [{"y": y, "z": z, "section": "bar"} for y in (-0.22, 0.22) for z in (-0.32, 0.32)]}
        mats = {"c": {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.0035,
                      "ft": 2.5e6, "Ets": 1.5e9, "rat": 0.1, "rho": 2500},
                "s": {"type": "steel02", "E": 2.0e11, "Fy": 4.0e8, "b": 0.01, "R0": 20.0, "rho": 7850}}
        extra_sec = {"bar": {"type": "bar", "material": "s", "shape": "circle", "diameter": 0.03}}
    else:
        girder = {"type": "beam", "material": "c", "shape": "rect", "width": 0.6, "height": 0.8}
        mats = {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2500}}
        extra_sec = {}
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines",
                  "points": {"A": [0, 0, 1], "M": [L / 2, 0, 1], "B": [L, 0, 1], "GA": [0, 0, 0], "GB": [L, 0, 0]},
                  "lines": {"g": {"through": ["A", "M", "B"], "size": 1.0},
                            "ba": {"through": ["A", "GA"], "size": 2.0},
                            "bb": {"through": ["B", "GB"], "size": 2.0}}},
         "materials": mats,
         "sections": dict(extra_sec, girder=girder,
                          pot={"type": "connector", "stiffness": [5e9, 2e8, 2e8, 1e8, 1e8, 1e6],
                               "connector_damping": [1e5, 1e5, 1e5, 0, 0, 0]},
                          slide={"type": "connector", "stiffness": [5e9, 0, 0, 1e8, 1e8, 1e6],
                                 "laws": {"uy": {"type": "friction", "mu": 0.05, "normal": "ux", "stiffness": 1e8},
                                          "uz": {"type": "friction", "mu": 0.05, "normal": "ux", "stiffness": 1e8}}}),
         "assign": [{"region": "g", "section": "girder", "orientation": [0, 1, 0]},
                    {"region": "ba", "section": "pot", "orientation": [1, 0, 0]},
                    {"region": "bb", "section": "slide", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "GA", "dofs": [1] * 6}, {"region": "GB", "dofs": [1] * 6}],
         "amplitudes": {"push": [[0, 0], [0.5, 1.0], [1.0, -0.3]]},
         "loads": [{"region": "all", "gravity": [0, 0, -9.80665], "amplitude": "hold"},
                   {"region": "M", "vector": [2e5, 5e4, 0], "amplitude": "push"},
                   {"ground_motion": {"path": str(rec), "direction": [1.0, 0.0, 0.0]}}],
         "solver": {"mode": "dynamic", "backend": "numba", "threads": threads, "total_time": total,
                    "damping": {"type": "rayleigh", "alpha": 0.5}},
         "output": {"path": str(tmp_path / f"{name}.h5"), "every": every, "fields": ["displacement"],
                    "monitors": {"ux": {"kind": "displacement", "region": "M", "dof": "x"},
                                 "RA": {"kind": "reaction", "region": "GA", "dof": "x"},
                                 "RB": {"kind": "reaction", "region": "GB", "dof": "x"}}}}
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
    for k in ("q", "v", "orient"):
        assert np.array_equal(getattr(s0, k), getattr(s1, k)), k
    assert np.array_equal(s0._conn_slip, s1._conn_slip) and np.array_equal(s0._conn_committed, s1._conn_committed)
    assert np.array_equal(s0.connector_deformation, s1.connector_deformation)
    assert np.array_equal(np.asarray(s0.end_actions), np.asarray(s1.end_actions))
    l0, l1 = s0.diagnostics["energy"], s1.diagnostics["energy"]
    assert (l0.external_work, l0.internal_work, l0.dissipated) == (l1.external_work, l1.internal_work, l1.dissipated)
    for k in o0.monitors:
        assert np.array_equal(o0.monitors[k], o1.monitors[k]), k
    _same_checkpoint(s0, s1)
    return s0, s1


def test_a_frame_girder_on_bearings_takes_the_compiled_step(tmp_path):
    s0, s1 = _same(_girder(tmp_path, "fg"), tmp_path, "fg")
    assert type(s1).__name__ == "MixedStructuralSolver"
    assert np.abs(s1._conn_slip).max() > 0.0                 # the sliding bearing slid


def test_a_fibre_girder_on_bearings_takes_the_compiled_step(tmp_path):
    s0, s1 = _same(_girder(tmp_path, "bg", fibre=True), tmp_path, "bg")
    assert type(s1).__name__ == "FiberBeamSolver" and s1._kernel is not None
    assert np.array_equal(s0._state_c, s1._state_c) and np.array_equal(s0._state_s, s1._state_s)
    assert np.array_equal(s0.actions_abs, s1.actions_abs)


def test_the_threaded_fibre_kernel_in_the_compiled_step_is_the_serial_one(tmp_path):
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    _o1, s1 = _run(_girder(tmp_path, "t1", fibre=True), True)
    _o2, s2 = _run(_girder(tmp_path, "t2", fibre=True, threads=2), True)
    assert s2.compiled_step_refusal is None
    _same_checkpoint(s1, s2)


def test_compiled_connectors_are_the_python_ones_to_round_off(tmp_path):
    """The step-by-step loop with the connectors compiled and with the NumPy
    reference (`backend: numpy`: the switch that turned the compiled
    connectors off went in v1.18): the same answer to the
    Jacobian's round-off."""
    d = _girder(tmp_path, "rc", total=0.02)
    ref = dict(d, solver=dict(d["solver"], backend="numpy"), output=dict(d["output"], path=str(tmp_path / "rc0.h5")))
    _o0, s0 = _run(ref, False)
    _o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "rc1.h5"))), False)
    assert np.abs(s0.q - s1.q).max() <= 1e-6 * np.abs(s0.q).max()


def test_a_fibre_restart_resumes_where_the_run_stopped(tmp_path):
    d = _girder(tmp_path, "rs", fibre=True, total=0.03, every=50)
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



def test_a_short_fibre_run_keeps_the_loop_unless_asked(tmp_path, monkeypatch):
    """Compiling the fibre step costs about 3 s once per process: a run too
    short to pay for it keeps the loop (the same answer) unless
    VFIFE_COMPILED_STEP=1 asks for it."""
    from vfife.kernels import numba_step
    monkeypatch.setattr(numba_step, "_FIBER_CHUNKS", {})
    monkeypatch.delenv("VFIFE_COMPILED_STEP", raising=False)
    box = {}
    run_case(_girder(tmp_path, "sh", fibre=True, total=0.002), on_ready=lambda s, p: box.setdefault("s", s))
    assert "short fibre run" in (box["s"].compiled_step_refusal or "")


def test_a_fibre_models_bearings_are_held_to_the_connector_rules(tmp_path):
    """A law on a component that has a stiffness, and friction pressed by a
    component with none: refused in a fibre model as in a mixed one (they
    ran, until the review of v1.4)."""
    from vfife.api import prepare_case
    d = _girder(tmp_path, "v1", fibre=True)
    d["sections"]["slide"]["stiffness"] = [5e9, 1e6, 0, 1e8, 1e8, 1e6]
    with pytest.raises(Exception, match="has a law AND a stiffness"):
        prepare_case(d)
    d = _girder(tmp_path, "v2", fibre=True)
    d["sections"]["slide"]["stiffness"] = [0, 0, 0, 1e8, 1e8, 1e6]
    with pytest.raises(Exception, match="plain linear stiffness"):
        prepare_case(d)
