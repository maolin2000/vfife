"""Moving loads and vehicles in the compiled step (1.6).

The compiled step refused every load group that was a callable.
The moving loads' axles and the vehicles are now evaluated
inside the step, by the compiled functions of kernels/numba_loads that the
step-by-step loop runs too (`LoadPath.shape`, `Vehicle.__call__`): one
arithmetic, so the two paths agree to the bit by construction. The
vehicles' state advances inside the chunk and is handed back with their
contact losses, on the run's clock, in step order.

Each case runs with `VFIFE_COMPILED_STEP` off and on and the state, the
monitors, the ledger, the vehicles' state and contact bookkeeping and the
members' end actions are compared with `np.array_equal`:

* a platoon of axles driving on and off a simply supported beam, with a
  fixed load group listed after it (the groups keep the loop's order);
* the same on members with released ends (the condensation);
* a two-axle vehicle that pitches, one axle on a tyre and one rigid, on a
  PSD road rough enough for the tyre to leave it;
* threads, and a restart from a stopped run mid-crossing;
* a moving load on a fibre girder (the fibre step);
* a callable of a script's own is still refused.
"""
import os

import h5py
import numpy as np
import pytest

from vfife.api import run_case
from vfife.core.diagnostics import AnalysisInterrupted

numba = pytest.importorskip("numba")

G = 9.80665
STEEL = {"type": "linear_elastic", "E": 2.1e11, "nu": 0.3, "rho": 7850.0}
SEC = {"type": "beam", "material": "steel", "shape": "general", "area": 0.02, "iy": 2.0e-4, "iz": 2.0e-4, "j": 4.0e-4}
PLATOON = {"axles": [[0.0, 5.0e4], [-3.0, 5.0e4], [-7.5, 3.0e4]], "start": -2.0, "speed": 60.0}
TRUCK = {"name": "truck", "speed": 25.0, "start": -1.0,
         "body": {"mass": 12000.0, "pitch_inertia": 4.0e4, "cg": -1.5},
         "axles": [{"offset": 0.0, "unsprung_mass": 500.0, "suspension": {"stiffness": 8.0e5, "damping": 2.0e4},
                    "tyre": {"stiffness": 3.0e6, "damping": 1.0e3}},
                   {"offset": -3.6, "suspension": {"stiffness": 1.2e6, "damping": 3.0e4}}]}


def _beam(tmp_path, name, loads, total, threads=1, releases=False, every=200, monitors=None, stiff=1.0):
    """0-20 m on ten members, pinned at both ends, with a hold gravity."""
    sec = dict(SEC, iy=SEC["iy"] * stiff, iz=SEC["iz"] * stiff, j=SEC["j"] * stiff)
    pts = {f"P{i}": [2.0 * i, 0.0, 0.0] for i in range(11)}
    lines = {"deck": {"through": [f"P{i}" for i in range(11)], "size": 2.0}}
    assign = [{"region": "deck", "section": "d", "orientation": [0, 1, 0]}]
    if releases:
        # a hinge at mid-span: members 4-5 (released at j) and 5-6
        pts = {f"P{i}": [2.0 * i, 0.0, 0.0] for i in range(11)}
        lines = {"west": {"through": [f"P{i}" for i in range(6)], "size": 2.0},
                 "east": {"through": [f"P{i}" for i in range(5, 11)], "size": 2.0}}
        assign = [{"region": "west", "section": "d", "orientation": [0, 1, 0], "releases": ["j_ry", "j_rz"]},
                  {"region": "east", "section": "d", "orientation": [0, 1, 0]}]
    mon = {"wM": {"kind": "displacement", "region": "P5", "dof": "z"},
           "RA": {"kind": "reaction", "region": "P0", "dof": "z"}}
    mon.update(monitors or {})
    return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "lines", "points": pts, "lines": lines},
            "materials": {"steel": STEEL}, "sections": {"d": sec},
            "assign": assign,
            "constraints": [{"region": "P0", "dofs": [1, 1, 1, 1, 0, 1]}, {"region": "P10", "dofs": [0, 1, 1, 1, 0, 1]}],
            "loads": loads,
            "solver": {"mode": "dynamic", "backend": "numba", "threads": threads, "total_time": total,
                       "damping": {"type": "viscous", "coeff": 0.5}},
            "output": {"path": str(tmp_path / f"{name}.h5"), "every": every,
                       "fields": ["displacement", "bending_y"], "monitors": mon}}


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
        assert np.array_equal(np.asarray(ca[k]), np.asarray(cb[k]), equal_nan=True), k


def _same_vehicles(s0, s1):
    v0, v1 = getattr(s0, "_vehicles", {}), getattr(s1, "_vehicles", {})
    assert set(v0) == set(v1)
    for name in v0:
        a, b = v0[name], v1[name]
        for f in ("x", "v", "a", "contact", "_surf_prev", "_out"):
            assert np.array_equal(np.asarray(getattr(a, f), float), np.asarray(getattr(b, f), float)), (name, f)
        assert float(a._t_done) == float(b._t_done)
        assert (a.lift_off, a.lift_tyre, a.tension_rigid, a.lift_first) == \
               (b.lift_off, b.lift_tyre, b.tension_rigid, b.lift_first), name
        if a.member_load is None:
            assert b.member_load is None
        else:
            assert np.array_equal(a.member_load[0], b.member_load[0])
            assert np.array_equal(a.member_load[1], b.member_load[1])
    d0, d1 = s0.diagnostics.get("vehicle_contact"), s1.diagnostics.get("vehicle_contact")
    assert d0 == d1


def _same(d, tmp_path, name, used=True):
    o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}0.h5"))), False)
    o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}1.h5"))), True)
    assert o0.status == o1.status == "COMPLETED_DYNAMIC", (o0.message, o1.message)
    if used:
        assert s1.compiled_step_refusal is None, s1.compiled_step_refusal
    for k in ("q", "v", "orient", "end_actions"):
        assert np.array_equal(getattr(s0, k), getattr(s1, k)), k
    l0, l1 = s0.diagnostics["energy"], s1.diagnostics["energy"]
    assert (l0.external_work, l0.internal_work, l0.dissipated) == (l1.external_work, l1.internal_work, l1.dissipated)
    for k in o0.monitors:
        assert np.array_equal(o0.monitors[k], o1.monitors[k]), k
    _same_vehicles(s0, s1)
    _same_checkpoint(s0, s1)
    with h5py.File(tmp_path / f"{name}0.h5") as f0, h5py.File(tmp_path / f"{name}1.h5") as f1:
        for step in f0["steps"]:
            for key in ("displacement", "bending_y"):
                if key in f0["steps"][step]:
                    assert np.array_equal(f0["steps"][step][key][...], f1["steps"][step][key][...]), (step, key)
    return o0, s0, s1


def _platoon(tmp_path, name, total=0.5, **kw):
    # the fixed gravity group AFTER the moving load: the groups are summed in
    # the loop's order, a moving one between fixed ones included
    return _beam(tmp_path, name, [{"region": "all", "moving_load": PLATOON},
                                  {"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}], total, **kw)


def test_a_platoon_drives_on_and_off(tmp_path):
    o, s0, s1 = _same(_platoon(tmp_path, "pl"), tmp_path, "pl")
    w = np.asarray(o.monitors["wM"])
    assert w.min() < -1e-3 and abs(w[-1]) < 0.5 * abs(w.min())      # crossed, and rang after


def test_a_platoon_on_released_members(tmp_path):
    _same(_platoon(tmp_path, "rel", releases=True), tmp_path, "rel")


def _truck_case(tmp_path, name, total=0.9, threads=1, every=200, roughness=None, tyres=False, stiff=1.0):
    veh = dict(TRUCK)
    if tyres:
        # both axles on tyres: nothing to pull the deck up
        veh["axles"] = [veh["axles"][0], dict(veh["axles"][1], unsprung_mass=400.0,
                                              tyre={"stiffness": 3.0e6, "damping": 1.0e3})]
    if roughness is not None:
        veh["roughness"] = roughness
    mon = {"heave": {"kind": "vehicle", "of": "truck", "field": "heave"},
           "pitch": {"kind": "vehicle", "of": "truck", "field": "pitch"},
           "P0": {"kind": "vehicle", "of": "truck", "field": "contact_force", "axle": 0},
           "P1": {"kind": "vehicle", "of": "truck", "field": "contact_force", "axle": 1}}
    return _beam(tmp_path, name, [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"},
                                  {"region": "all", "vehicle": veh}],
                 total, threads=threads, every=every, monitors=mon, stiff=stiff)


ROUGH = {"psd": {"g0": 2.0e-3, "n0": 0.1, "w": 2.0, "n_min": 0.05, "n_max": 2.0, "count": 60, "seed": 3}}


def test_a_pitching_truck_on_a_rough_road_loses_a_tyre(tmp_path):
    o, s0, s1 = _same(_truck_case(tmp_path, "tr", roughness=ROUGH), tmp_path, "tr")
    veh = s1._vehicles["truck"]
    assert veh.lift_off > 0 and np.isfinite(veh.lift_first)
    assert s1.diagnostics["vehicle_contact"]["truck"]["tyre_lift_off_steps"] == veh.lift_tyre
    P0 = np.asarray(o.monitors["P0"])
    assert P0.min() <= 0.0 and P0.max() > 5.0e4


def test_a_truck_on_tyres_on_a_smooth_road_keeps_contact(tmp_path):
    o, s0, s1 = _same(_truck_case(tmp_path, "sm", total=0.6, tyres=True, stiff=20.0), tmp_path, "sm")
    assert s1._vehicles["truck"].lift_off == 0
    assert "vehicle_contact" not in s1.diagnostics


def test_threads_are_the_serial_answer_under_a_truck(tmp_path):
    d1 = _truck_case(tmp_path, "th1", total=0.4, roughness=ROUGH)
    d2 = _truck_case(tmp_path, "th2", total=0.4, threads=2, roughness=ROUGH)
    _o1, s1 = _run(d1, True)
    _o2, s2 = _run(d2, True)
    assert s1.compiled_step_refusal is None and s2.compiled_step_refusal is None
    assert np.array_equal(s1.q, s2.q) and np.array_equal(s1.v, s2.v)
    _same_vehicles(s1, s2)


def test_a_restart_mid_crossing_resumes_where_the_run_stopped(tmp_path):
    d = _truck_case(tmp_path, "rs", total=0.6, roughness=ROUGH)
    _w, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "w.h5"))), True, progress_every=300)
    ends = {}
    for compiled in (False, True):
        path = str(tmp_path / f"p{compiled}.h5")
        calls = {"n": 0}

        def stop(*_a):
            calls["n"] += 1
            return calls["n"] < 3

        with pytest.raises(AnalysisInterrupted):
            _run(dict(d, output=dict(d["output"], path=path)), compiled, on_progress=stop, progress_every=300)
        rest, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"r{compiled}.h5"))), compiled,
                        restart=path, progress_every=300)
        assert rest.status == "COMPLETED_DYNAMIC"
        ends[compiled] = sr
    _same_checkpoint(ends[False], ends[True])
    _same_checkpoint(ends[True], sw)
    _same_vehicles(ends[False], ends[True])


def test_a_callable_of_a_scripts_own_is_refused(tmp_path):
    from vfife.core import compiled_step
    d = _platoon(tmp_path, "cb", total=0.01)
    _o, s = _run(d, True)
    groups = [(a, (lambda p, t, out: out) if callable(F) else F) for a, F in s.step_data(0)[3]]
    fixed, moving, prescribed, _g, ground = s.step_data(0)
    assert compiled_step.refusal(s, fixed, moving, prescribed, groups, ground) == \
        "a load group that is a callable of its own"


# ---------------------------------------------------------------- fibre step
def _fibre_crossing(tmp_path, name, total=0.3, every=100):
    L = 8.0
    sec = {"type": "fiber", "material": "c", "shape": "rect", "width": 0.3, "height": 0.5,
           "ny": 2, "nz": 6, "integration": "lobatto3",
           "bars": [{"y": y, "z": z, "section": "bar"} for y in (-0.1, 0.1) for z in (-0.2, 0.2)]}
    return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "lines",
                     "points": {"A": [0, 0, 0], "M": [L / 2, 0, 0], "B": [L, 0, 0]},
                     "lines": {"g": {"through": ["A", "M", "B"], "size": 1.0}}},
            "materials": {"c": {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.0035,
                                "ft": 2.5e6, "Ets": 1.5e9, "rat": 0.1, "rho": 2500},
                          "s": {"type": "steel02", "E": 2.0e11, "Fy": 4.0e8, "b": 0.01, "R0": 20.0, "rho": 7850}},
            "sections": {"girder": sec, "bar": {"type": "bar", "material": "s", "shape": "circle", "diameter": 0.02}},
            "assign": [{"region": "g", "section": "girder", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "A", "dofs": [1, 1, 1, 1, 0, 1]}, {"region": "B", "dofs": [0, 1, 1, 1, 0, 1]}],
            "loads": [{"region": "g", "moving_load": {"axles": [[0.0, 1.0e5], [-2.0, 8.0e4]], "start": -1.0,
                                                      "speed": 40.0}},
                      {"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
            "solver": {"mode": "dynamic", "backend": "numba", "total_time": total,
                       "damping": {"type": "viscous", "coeff": 2.0}},
            "output": {"path": str(tmp_path / f"{name}.h5"), "every": every,
                       "fields": ["displacement"],
                       "monitors": {"wM": {"kind": "displacement", "region": "M", "dof": "z"}}}}


def test_a_moving_load_crosses_a_fibre_girder(tmp_path):
    old = os.environ.get("VFIFE_COMPILED_STEP")
    try:
        d = _fibre_crossing(tmp_path, "fc")
        o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "fc0.h5"))), False)
        os.environ["VFIFE_COMPILED_STEP"] = "1"        # a short run: asked for
        o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "fc1.h5"))), True)
    finally:
        if old is None:
            os.environ.pop("VFIFE_COMPILED_STEP", None)
        else:
            os.environ["VFIFE_COMPILED_STEP"] = old
    assert o0.status == o1.status == "COMPLETED_DYNAMIC", (o0.message, o1.message)
    assert s1.compiled_step_refusal is None, s1.compiled_step_refusal
    for k in ("q", "v", "orient", "_state_c", "_state_s"):
        assert np.array_equal(getattr(s0, k), getattr(s1, k)), k
    for k in o0.monitors:
        assert np.array_equal(o0.monitors[k], o1.monitors[k]), k
    assert np.asarray(o0.monitors["wM"]).min() < -1e-4
