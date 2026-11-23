"""Plastic hinges, breaking members and split-off fragments in the compiled
step (1.6).

The compiled step took no model with members that can break.
It now runs the hinge update and the failure criterion inside the step, and
a member that reaches its capacity is an EVENT: the kernel stops before that
step, the loop takes it -- marks the break, deactivates or splits the member,
parks what has no body left -- and the compiled step is rebuilt on what is
left. Parked particles and appendages are carried inside the step. The
hinge update and the carrying are ONE compiled function for both paths, so
the loop and the compiled step agree to the bit by construction.

Each case runs with `VFIFE_COMPILED_STEP` off and on and the state, the
failure events, the capacity ratios, the plastic rotations, the ledger and
the monitors are compared with `np.array_equal`:

* a weight dropped onto a cable that breaks on the overshoot (a deactivated
  bar), and the same cable removed at a time;
* a cantilever whose root turns into a plastic hinge and then tears on
  rotation, the piece swinging away and falling (a split with its mass);
* a girder on two hangers losing one hanger, with a massless crossbeam on
  the node left behind (a parked particle and an appendage);
* threads, and a restart from a stopped run across a break;
* a frame that breaks under its first step's forces.
"""
import os

import h5py
import numpy as np
import pytest

from test_member_failure import G, STEEL, _write_msh
from vfife.api import run_case
from vfife.core.diagnostics import AnalysisInterrupted

numba = pytest.importorskip("numba")


def _hanging(tmp_path, name, strength_over_static=1.5, removal=None, total=0.12, threads=1):
    _write_msh(tmp_path / f"{name}.msh",
               [(0, 0, 0), (1, 0, 0), (1, 0, -1)],
               {"root": (0, [0]), "tip": (0, [1]), "weight": (0, [2]),
                "arm": (1, [(0, 1)]), "cable": (1, [(1, 2)])})
    A = 1e-4
    T_static = 1000.0 * G
    d = {"name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / f"{name}.msh")},
         "materials": {"steel": STEEL},
         "sections": {"arm": {"type": "beam", "material": "steel", "shape": "general",
                              "area": 0.1, "iy": 1e-2, "iz": 1e-2, "j": 1e-2},
                      "cable": {"type": "bar", "material": "steel", "shape": "general", "area": A}},
         "assign": [{"region": "arm", "section": "arm", "orientation": [0, 0, 1]},
                    {"region": "cable", "section": "cable", "tension_only": True,
                     "failure": {"tensile_strength": strength_over_static * T_static / A}}],
         "nodal_masses": [{"region": "weight", "mass": 1000.0}],
         "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]},
                         {"region": "weight", "dofs": [1, 1, 0, 1, 1, 1]}],
         "loads": [{"region": "weight", "vector": [0, 0, -T_static], "amplitude": "hold"}],
         "solver": {"backend": "numba", "threads": threads, "mode": "dynamic", "total_time": total,
                    "damping": {"type": "viscous", "coeff": 0.5}},
         "output": {"path": str(tmp_path / f"{name}.h5"), "fields": ["displacement", "capacity_ratio"], "every": 50,
                    "monitors": {"w": {"kind": "displacement", "region": "weight", "dof": "z"},
                                 "R": {"kind": "reaction", "region": "root", "dof": "z"}}}}
    if removal is not None:
        d["removals"] = [{"region": "cable", "time": removal}]
    return d


FY, C, I, A = 3.0e8, 0.1, 1e-5, 1e-2
M_P = FY * I / C
L = 2.0


def _hinged_cantilever(tmp_path, name, total=0.6, theta_u=0.02, threads=1, first_step=False):
    n = 4
    xs = np.linspace(0, L, n + 1)
    _write_msh(tmp_path / f"{name}.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [n]),
                "first": (1, [(0, 1)]), "rest": (1, [(k, k + 1) for k in range(1, n)])})
    beam = {"type": "beam", "material": "steel", "shape": "general", "area": A, "iy": I, "iz": I, "j": 2 * I}
    fail = {"yield_strength": FY, "extreme_fibre": [C, C], "rotation_capacity": theta_u}
    P = 1.5 * M_P / L
    amp = [[0, 1.0], [1.0, 1.0]] if first_step else [[0, 0], [0.5, 1.0], [1.0, 1.0]]
    return {"name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": str(tmp_path / f"{name}.msh")},
            "materials": {"steel": STEEL},
            "sections": {"b": beam},
            "assign": [{"region": "first", "section": "b", "orientation": [0, 0, 1], "failure": fail},
                       {"region": "rest", "section": "b", "orientation": [0, 0, 1]}],
            "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
            "amplitudes": {"a": amp},
            "loads": [{"region": "tip", "vector": [0, 0, -P], "amplitude": "a"},
                      {"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
            "solver": {"backend": "numba", "threads": threads, "mode": "dynamic", "total_time": total,
                       "damping": {"type": "viscous", "coeff": 50.0}},
            "output": {"path": str(tmp_path / f"{name}.h5"), "every": 100,
                       "fields": ["displacement", "bending_y", "capacity_ratio", "hinge_rotation", "member_failed"],
                       "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}}}


def _girder_on_hangers(tmp_path, name, total=0.3, threads=1):
    """A girder hung from two points of a stiff frame by two cables, a
    massless crossbeam (rho ~ 0) sticking out of the girder's left node, the
    left hanger's lower half cut at 0.05 s: the girder swings on the right
    one, the crossbeam goes with it as an appendage; the hanger's joint,
    left with the massless upper half alone, is parked."""
    _write_msh(tmp_path / f"{name}.msh",
               [(0, 0, 5), (4, 0, 5), (0, 0, 0), (2, 0, 0), (4, 0, 0), (0, 1, 0), (0.5, 0, 5), (0, 0, 3)],
               {"anchors": (0, [0, 1]), "girder": (1, [(2, 3), (3, 4)]), "left": (1, [(7, 2)]),
                "left_up": (1, [(0, 7)]),
                "right": (1, [(1, 4)]), "cross": (1, [(2, 5)]), "stub": (1, [(0, 6)]),
                "gleft": (0, [2]), "top_left": (0, [0]), "joint": (0, [7])})
    return {"name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": str(tmp_path / f"{name}.msh")},
            "materials": {"steel": STEEL, "light": dict(STEEL, rho=1e-6)},
            "sections": {"g": {"type": "beam", "material": "steel", "shape": "rect", "width": 0.3, "height": 0.5},
                         "x": {"type": "beam", "material": "light", "shape": "rect", "width": 0.1, "height": 0.1},
                         "h": {"type": "bar", "material": "steel", "shape": "general", "area": 2e-3},
                         "hl": {"type": "bar", "material": "light", "shape": "general", "area": 2e-3}},
            "assign": [{"region": "girder", "section": "g", "orientation": [0, 1, 0]},
                       {"region": "cross", "section": "x", "orientation": [0, 0, 1]},
                       {"region": "stub", "section": "x", "orientation": [0, 1, 0]},
                       {"region": "left", "section": "h", "tension_only": True},
                       {"region": "left_up", "section": "hl", "tension_only": True},
                       {"region": "right", "section": "h", "tension_only": True}],
            "removals": [{"region": "left", "time": 0.05}],
            "constraints": [{"region": "anchors", "dofs": [1, 1, 1, 1, 1, 1]}],
            "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
            "solver": {"backend": "numba", "threads": threads, "mode": "dynamic", "total_time": total,
                       "damping": {"type": "viscous", "coeff": 1.0},
                       # the light crossbeam alone would set a step of 3e-10 s
                       "mass_scaling": {"target_dt": 2e-5, "max_added_fraction": 0.05}},
            "output": {"path": str(tmp_path / f"{name}.h5"), "every": 500,
                       "fields": ["displacement", "member_failed"],
                       "monitors": {"wl": {"kind": "displacement", "region": "gleft", "dof": "z"}}}}


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


def _same(d, tmp_path, name, used=True):
    o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}0.h5"))), False)
    o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}1.h5"))), True)
    assert o0.status == o1.status == "COMPLETED_DYNAMIC", (o0.message, o1.message)
    if used:
        assert s1.compiled_step_refusal is None, s1.compiled_step_refusal
    for k in ("q", "v", "orient", "mass", "physical_mass"):
        assert np.array_equal(getattr(s0, k), getattr(s1, k)), k
    assert np.array_equal(s0.breaking.member_active, s1.breaking.member_active)
    assert np.array_equal(s0.breaking.end_broken, s1.breaking.end_broken)
    assert np.array_equal(np.asarray(s0.breaking.failure_events), np.asarray(s1.breaking.failure_events))
    assert np.array_equal(s0.breaking.capacity_ratio, s1.breaking.capacity_ratio, equal_nan=True)
    assert np.array_equal(s0.mesh.elements, s1.mesh.elements)
    if getattr(s0, "_plastic_d", None) is not None:
        assert np.array_equal(s0._plastic_d, s1._plastic_d)
        assert np.array_equal(s0._hinge_rotation, s1._hinge_rotation)
    l0, l1 = s0.diagnostics["energy"], s1.diagnostics["energy"]
    assert (l0.external_work, l0.internal_work, l0.dissipated) == (l1.external_work, l1.internal_work, l1.dissipated)
    for k in o0.monitors:
        assert np.array_equal(o0.monitors[k], o1.monitors[k]), k
    _same_checkpoint(s0, s1)
    with h5py.File(tmp_path / f"{name}0.h5") as f0, h5py.File(tmp_path / f"{name}1.h5") as f1:
        for key in ("displacement", "member_failed", "capacity_ratio", "hinge_rotation"):
            last = sorted(f0["steps"])[-1]
            if key in f0["steps"][last]:
                assert np.array_equal(f0["steps"][last][key][...], f1["steps"][last][key][...], equal_nan=True), key
    return o0, s0, s1


def test_a_cable_that_breaks_on_the_overshoot(tmp_path):
    o, s0, s1 = _same(_hanging(tmp_path, "cb"), tmp_path, "cb")
    ev = np.asarray(s1.breaking.failure_events)
    assert ev.shape == (1, 4) and ev[0, 3] == 0.0 and 1.0 <= ev[0, 2] < 1.05
    assert np.asarray(o.monitors["w"])[-1] < -0.02                     # falling


def test_a_cable_removed_at_its_time(tmp_path):
    o, s0, s1 = _same(_hanging(tmp_path, "cr", strength_over_static=100.0, removal=0.05, total=0.2),
                      tmp_path, "cr")
    ev = np.asarray(s1.breaking.failure_events)
    assert ev.shape == (1, 4) and ev[0, 3] == 1.0 and ev[0, 0] == pytest.approx(0.05, abs=2e-4)


def test_a_hinged_cantilever_that_tears_and_falls(tmp_path):
    o, s0, s1 = _same(_hinged_cantilever(tmp_path, "hc"), tmp_path, "hc")
    assert np.abs(s1._plastic_d).max() > 0.0                            # the hinge rotated
    ev = np.asarray(s1.breaking.failure_events)
    assert len(ev) >= 1 and s1.breaking.end_broken.any()                 # and tore
    assert np.asarray(o.monitors["w"])[-1] < -0.5                       # the piece fell


def test_a_frame_that_breaks_in_its_first_step(tmp_path):
    _same(_hinged_cantilever(tmp_path, "fs", total=0.3, theta_u=1e-6, first_step=True), tmp_path, "fs")


def test_a_hanger_cut_leaves_an_appendage_and_a_parked_joint(tmp_path):
    o, s0, s1 = _same(_girder_on_hangers(tmp_path, "gh"), tmp_path, "gh")
    assert len(s1._appendages) == 2 and len(s1._parked) == 1
    assert np.asarray(o.monitors["wl"])[-1] < -0.2                      # the girder swung down


def test_threads_are_the_serial_answer_across_a_break(tmp_path):
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    _o1, s1 = _run(_hinged_cantilever(tmp_path, "t1", total=0.4), True)
    _o2, s2 = _run(_hinged_cantilever(tmp_path, "t2", total=0.4, threads=2), True)
    assert s2.compiled_step_refusal is None
    _same_checkpoint(s1, s2)


def test_a_restart_across_a_break_resumes_where_the_run_stopped(tmp_path):
    d = _hanging(tmp_path, "rs", total=0.12)
    _w, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "w.h5"))), True, progress_every=200)
    ends = {}
    for compiled in (False, True):
        path = str(tmp_path / f"p{compiled}.h5")
        calls = {"n": 0}

        def stop(*_a):
            calls["n"] += 1
            return calls["n"] < 3

        with pytest.raises(AnalysisInterrupted):
            _run(dict(d, output=dict(d["output"], path=path)), compiled, on_progress=stop, progress_every=200)
        rest, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"r{compiled}.h5"))), compiled,
                        restart=path, progress_every=200)
        assert rest.status == "COMPLETED_DYNAMIC"
        ends[compiled] = sr
    _same_checkpoint(ends[False], ends[True])
    _same_checkpoint(ends[True], sw)


def test_260_a_restart_after_a_hanger_is_cut_keeps_the_parked_joint_and_the_appendage(tmp_path):
    """The parked particles and the appendages were not in the frame
    solvers' checkpoint: a restart after the cut let the hanger's joint and
    the crossbeam's tip go free, and the girder ended 3 cm from where the
    uninterrupted run left it."""
    d = _girder_on_hangers(tmp_path, "pk", total=0.15)
    _w, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "w.h5"))), False, progress_every=500)
    assert sw._parked and sw._appendages
    path = str(tmp_path / "p.h5")
    calls = {"n": 0}

    def stop(*_a):
        calls["n"] += 1
        return calls["n"] < 8                  # t = 0.08 s, after the cut at 0.05 s

    with pytest.raises(AnalysisInterrupted):
        _run(dict(d, output=dict(d["output"], path=path)), False, on_progress=stop, progress_every=500)
    rest, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "r.h5"))), False,
                    restart=path, progress_every=500)
    assert rest.status == "COMPLETED_DYNAMIC"
    assert sorted(sr._parked) == sorted(sw._parked)
    assert [ap["tips"] for ap in sr._appendages] == [ap["tips"] for ap in sw._appendages]
    _same_checkpoint(sr, sw)


def test_260_a_restart_after_a_park_without_spares_keeps_the_particle_parked(tmp_path):
    """Both hangers of the left side removed without a split: the joint
    between them connects nothing and is parked, with no spare particles in
    the model. A restart let it go -- |dq| reached 1e18 (the review of
    v1.19)."""
    d = _girder_on_hangers(tmp_path, "ns", total=0.15)
    d["removals"] = [{"region": "left", "time": 0.05, "split": False},
                     {"region": "left_up", "time": 0.05, "split": False}]
    for a in d["assign"]:
        if a["region"] == "left":
            a["section"] = "hl"
    _w, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "w.h5"))), False, progress_every=500)
    assert sw.bonds is None and sw._parked
    path = str(tmp_path / "p.h5")
    calls = {"n": 0}

    def stop(*_a):
        calls["n"] += 1
        return calls["n"] < 8

    with pytest.raises(AnalysisInterrupted):
        _run(dict(d, output=dict(d["output"], path=path)), False, on_progress=stop, progress_every=500)
    _r, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "r.h5"))), False,
                  restart=path, progress_every=500)
    assert np.array_equal(sr.q, sw.q) and np.array_equal(sr.v, sw.v)
    # (its `physical_mass` is not in a checkpoint without spares, and comes
    # back as it was at the setup; the particle is held, `mass` is the
    # loop's, and the step's gravity is the one the uninterrupted run built)
    _same_checkpoint(sr, sw)


@pytest.mark.parametrize("steps", [(0.03, 0.12), (0.03, 0.05, 0.07)])
def test_261_a_dynamic_step_after_a_park_and_its_restart(tmp_path, steps):
    """A dynamic step that began after the hanger was cut took its stable
    step from the parked joint and the crossbeam's tips, which have no mass
    of their own, and never ended; a restart in the step the cut was in did
    the same, from the masses the cut had changed. The
    carried particles are out of the bound, and a restart keeps the step the
    run was on."""
    d = _girder_on_hangers(tmp_path, "st", total=0.15)
    d["steps"] = [{"name": f"s{i}", "procedure": "dynamic", "total_time": t} for i, t in enumerate(steps)]
    out, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "w.h5"))), False, progress_every=500)
    assert out.status == "COMPLETED_DYNAMIC" and sw._parked
    assert sw.step < 8000
    path = str(tmp_path / "p.h5")
    calls = {"n": 0}

    def stop(*_a):
        calls["n"] += 1
        return calls["n"] < 10                 # in the step the cut (0.05 s) is in, after it

    with pytest.raises(AnalysisInterrupted):
        _run(dict(d, output=dict(d["output"], path=path)), False, on_progress=stop, progress_every=500)
    rest, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "r.h5"))), False,
                    restart=path, progress_every=500)
    assert rest.status == "COMPLETED_DYNAMIC" and sr.step == sw.step
    assert np.array_equal(sr.q, sw.q) and np.array_equal(sr.v, sw.v)


# ------------------------------------------------------------- fibre fracture
def _fibre_girder(tmp_path, name, total=0.4, threads=1, rupture=0.004, every=100):
    """A fibre RC girder on two piers with a heavy mid-span mass, under its
    weight and a growing point load: the bottom bars rupture at mid-span
    (a small rupture strain), the section fractures, and the two halves
    split off and fall."""
    L = 8.0
    sec = {"type": "fiber", "material": "c", "shape": "rect", "width": 0.3, "height": 0.5,
           "ny": 2, "nz": 6, "integration": "lobatto3",
           "bars": [{"y": y, "z": z, "section": "bar"} for y in (-0.1, 0.1) for z in (-0.2, 0.2)],
           "fracture": {"threshold": 0.2, "split": True}}
    return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "lines",
                     "points": {"A": [0, 0, 0], "M": [L / 2, 0, 0], "B": [L, 0, 0]},
                     "lines": {"g": {"through": ["A", "M", "B"], "size": 1.0}}},
            "materials": {"c": {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.0035,
                                "ft": 2.5e6, "Ets": 1.5e9, "rat": 0.1, "rho": 2500},
                          "s": {"type": "steel02", "E": 2.0e11, "Fy": 4.0e8, "b": 0.01, "R0": 20.0, "rho": 7850,
                                "rupture_strain": rupture}},
            "sections": {"girder": sec, "bar": {"type": "bar", "material": "s", "shape": "circle", "diameter": 0.02}},
            "assign": [{"region": "g", "section": "girder", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "A", "dofs": [1, 1, 1, 1, 0, 1]}, {"region": "B", "dofs": [0, 1, 1, 1, 0, 1]}],
            "amplitudes": {"push": [[0, 0], [1.0, 1.0]]},
            "loads": [{"region": "all", "gravity": [0, 0, -9.80665], "amplitude": "hold"},
                      {"region": "M", "vector": [0, 0, -4.0e5], "amplitude": "push"}],
            "solver": {"mode": "dynamic", "backend": "numba", "threads": threads, "total_time": total,
                       "damping": {"type": "viscous", "coeff": 2.0}},
            "output": {"path": str(tmp_path / f"{name}.h5"), "every": every,
                       "fields": ["displacement", "fractured", "member_failed"],
                       "monitors": {"wM": {"kind": "displacement", "region": "M", "dof": "z"}}}}


def _same_fibre(d, tmp_path, name):
    o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}0.h5"))), False)
    o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"{name}1.h5"))), True)
    assert o0.status == o1.status == "COMPLETED_DYNAMIC", (o0.message, o1.message)
    assert s1.compiled_step_refusal is None, s1.compiled_step_refusal
    for k in ("q", "v", "orient", "mass", "physical_mass", "_state_c", "_state_s", "_active"):
        assert np.array_equal(getattr(s0, k), getattr(s1, k)), k
    assert np.array_equal(s0.fracture.latch.point, s1.fracture.latch.point)
    assert np.array_equal(s0.fracture.latch.fractured, s1.fracture.latch.fractured)
    assert np.array_equal(np.asarray(s0.fracture.events, float), np.asarray(s1.fracture.events, float))
    assert np.array_equal(np.asarray(s0.breaking.failure_events), np.asarray(s1.breaking.failure_events))
    assert np.array_equal(s0.mesh.elements, s1.mesh.elements)
    l0, l1 = s0.diagnostics["energy"], s1.diagnostics["energy"]
    assert (l0.external_work, l0.internal_work, l0.dissipated) == (l1.external_work, l1.internal_work, l1.dissipated)
    for k in o0.monitors:
        assert np.array_equal(o0.monitors[k], o1.monitors[k]), k
    _same_checkpoint(s0, s1)
    return o0, s0, s1


def test_a_fibre_girder_fractures_and_its_halves_fall(tmp_path):
    o, s0, s1 = _same_fibre(_fibre_girder(tmp_path, "ff"), tmp_path, "ff")
    assert s1.fracture.latch.fractured.any()
    assert len(s1.splits) > 0                                 # the halves split off
    assert np.asarray(o.monitors["wM"])[-1] < -0.3           # and fell


def test_a_fibre_member_removed_at_its_time(tmp_path):
    d = _fibre_girder(tmp_path, "fr", rupture=0.5, total=0.2)
    d["removals"] = [{"region": "g", "time": 0.05}]
    o, s0, s1 = _same_fibre(d, tmp_path, "fr")
    ev = np.asarray(s1.breaking.failure_events)
    assert len(ev) == 8 and (ev[:, 3] == 1.0).all()


def test_fibre_threads_are_the_serial_answer_across_a_fracture(tmp_path):
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    _o1, s1 = _run(_fibre_girder(tmp_path, "ft1"), True)
    _o2, s2 = _run(_fibre_girder(tmp_path, "ft2", threads=2), True)
    assert s2.compiled_step_refusal is None and s1.fracture.latch.fractured.any()
    _same_checkpoint(s1, s2)


def test_a_fibre_restart_across_a_fracture_resumes_where_the_run_stopped(tmp_path):
    d = _fibre_girder(tmp_path, "frs", total=0.3, every=50)
    _w, sw = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "w.h5"))), True, progress_every=100)
    assert sw.fracture.latch.fractured.any()
    ends = {}
    for compiled in (False, True):
        path = str(tmp_path / f"p{compiled}.h5")
        calls = {"n": 0}

        def stop(*_a):
            calls["n"] += 1
            return calls["n"] < 3

        with pytest.raises(AnalysisInterrupted):
            _run(dict(d, output=dict(d["output"], path=path)), compiled, on_progress=stop, progress_every=100)
        rest, sr = _run(dict(d, output=dict(d["output"], path=str(tmp_path / f"r{compiled}.h5"))), compiled,
                        restart=path, progress_every=100)
        assert rest.status == "COMPLETED_DYNAMIC"
        ends[compiled] = sr
    _same_checkpoint(ends[False], ends[True])
    _same_checkpoint(ends[True], sw)


def test_fracture_with_two_fibre_sections_and_a_bearing(tmp_path):
    """A connector has no fibres: its row of the latch's denominators was 0,
    which the latch refuses, and the run stopped at its first step with
    "every element is out of at least one thing" (found in the review of
    v1.12). Two fibre sections make the denominators per member."""
    d = _fibre_girder(tmp_path, "bear", total=0.14)
    d["sections"]["girder"]["fracture"]["split"] = False
    d["sections"]["other"] = dict(d["sections"]["girder"], width=0.35)
    d["mesh"]["points"].update({"R1": [4.5, -1.0, -2.0], "R2": [4.5, 1.0, -2.0], "GB": [8.0, 0, -1.0]})
    d["mesh"]["lines"]["rider"] = {"through": ["R1", "R2"], "size": 0.5}
    d["mesh"]["lines"]["bb"] = {"through": ["B", "GB"], "size": 2.0}
    d["sections"]["slide"] = {"type": "connector", "stiffness": [5e9, 1e8, 1e8, 1e6, 1e8, 1e6]}
    d["assign"] += [{"region": "rider", "section": "other", "orientation": [0, 0, 1]},
                    {"region": "bb", "section": "slide", "orientation": [1, 0, 0]}]
    d["constraints"] = [c for c in d["constraints"] if c["region"] != "B"] + [
        {"region": "GB", "dofs": [1] * 6}, {"region": "B", "dofs": [0, 1, 0, 1, 0, 1]},
        {"region": "R1", "dofs": [1] * 6}, {"region": "R2", "dofs": [1] * 6}]
    d["loads"][1]["vector"] = [0, 0, -2.0e6]
    o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "bear0.h5"))), False)
    o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "bear1.h5"))), True)
    assert o0.status == o1.status == "COMPLETED_DYNAMIC", (o0.message, o1.message)
    assert s1.compiled_step_refusal is None, s1.compiled_step_refusal
    assert len(np.flatnonzero(s1.fracture.latch.fractured)) >= 1
    conn = np.flatnonzero(~s1._is_fiber.astype(bool))
    assert len(conn) == 1 and not s1.fracture.latch.fractured[conn].any()
    for k in ("q", "v", "_state_s", "_active"):
        assert np.array_equal(getattr(s0, k), getattr(s1, k)), k
    assert np.array_equal(np.asarray(s0.fracture.events, float), np.asarray(s1.fracture.events, float))


@pytest.mark.parametrize("kind", ["cable", "fibre"])
def test_191_a_break_does_not_depend_on_where_the_frames_and_the_checks_fall(tmp_path, kind):
    """An output frame or a progress check after step n evaluates the forces
    of the state step n + 1 starts from. A member at its capacity there was
    broken by that reading -- one step before its own step would have, so
    it carried nothing in the step it should have carried its last force
    in, and the event was stamped a step early. With a frame after every
    step every break was found that way; with none, none was: the end state
    depended on `output.every` and `progress_every` (the fibre girder:
    1.3 mm; found in the review of v1.12). A reading only reads."""
    def make(name, every):
        if kind == "cable":
            d = _hanging(tmp_path, name, total=0.12)
        else:
            d = _fibre_girder(tmp_path, name, total=0.12)
            d["loads"][1]["vector"] = [0, 0, -2.0e6]
        d["output"]["every"] = every
        return d
    ends = []
    for every, progress, compiled in ((1, 1, False), (10 ** 6, 10 ** 6, False), (7, 13, False), (7, 13, True)):
        d = make(f"{kind}{every}{int(compiled)}", every)
        box = {}
        old = os.environ.get("VFIFE_COMPILED_STEP")
        os.environ["VFIFE_COMPILED_STEP"] = "1" if compiled else "0"
        try:
            run_case(d, on_ready=lambda s, p: box.setdefault("s", s), progress_every=progress)
        finally:
            os.environ.pop("VFIFE_COMPILED_STEP", None) if old is None else os.environ.update(VFIFE_COMPILED_STEP=old)
        s = box["s"]
        events = np.asarray(s.breaking.failure_events, float)
        assert len(events) >= 1, "nothing broke"
        ends.append((s.q.copy(), s.v.copy(), events))
    for q, v, events in ends[1:]:
        assert np.array_equal(events, ends[0][2])
        assert np.array_equal(q, ends[0][0]) and np.array_equal(v, ends[0][1])


def test_191_a_restart_keeps_the_stress_a_member_broke_at(tmp_path):
    """A fractured member is not evaluated again: its fibres' stresses stay
    the ones it broke at, and that is what `fiber_stress` reports afterwards.
    They were not in the checkpoint, and a restarted run reported 0 for
    every broken member (404 MPa in the uninterrupted one)."""
    def make(name):
        d = _fibre_girder(tmp_path, name, total=0.1, every=200)
        d["sections"]["girder"]["fracture"]["split"] = False
        d["loads"][1]["vector"] = [0, 0, -2.0e6]
        d["output"]["fields"] = ["displacement", "fractured", "fiber_stress"]
        return d
    _o, sw = _run(make("whole"), True)
    n = {"k": 0}

    def stop(*_a):
        n["k"] += 1
        return n["k"] < 9
    with pytest.raises(AnalysisInterrupted):
        _run(make("part"), True, on_progress=stop, progress_every=int(sw.n_steps * 0.1))
    _r, sr = _run(make("rest"), True, restart=str(tmp_path / "part.h5"))
    broken = np.flatnonzero(sw.fracture.latch.fractured)
    assert len(broken) >= 1 and np.abs(sw.fiber_stress_rows()[broken]).max() > 1.0e8
    with h5py.File(tmp_path / "part.h5") as f:
        last = sorted(f["steps"])[-1]
        assert np.asarray(f["steps"][last]["fractured"]).any(), "the stop fell after a fracture"
    assert np.array_equal(sr.fiber_stress_rows(), sw.fiber_stress_rows())
    with h5py.File(tmp_path / "whole.h5") as a, h5py.File(tmp_path / "rest.h5") as b:
        frames = sorted(set(a["steps"]) & set(b["steps"]))
        assert len(frames) >= 2
        for k in frames:
            assert np.array_equal(np.asarray(a["steps"][k]["fiber_stress"]), np.asarray(b["steps"][k]["fiber_stress"])), k


def test_191_a_bar_of_a_truss_model_is_broken_by_its_step_not_by_a_frame(tmp_path):
    """The standalone truss solver had the same fault (found in the review
    of v1.12): with a frame after every step the bar carried nothing
    in its own last step, and the weight ended 0.1 mm from where it ends
    with no frame."""
    from test_member_failure import _write_msh as write
    ends = []
    for every in (1, 7, 10 ** 6):
        write(tmp_path / f"t{every}.msh", [(0, 0, 0), (0, 0, -1)],
              {"top": (0, [0]), "weight": (0, [1]), "bar": (1, [(0, 1)])})
        A_ = 1e-4; T = 1000.0 * G
        d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
             "mesh": {"type": "gmsh", "path": str(tmp_path / f"t{every}.msh")},
             "materials": {"steel": STEEL},
             "sections": {"bar": {"type": "bar", "material": "steel", "shape": "general", "area": A_}},
             "assign": [{"region": "all", "section": "bar", "tension_only": True,
                         "failure": {"tensile_strength": 1.5 * T / A_}}],
             "nodal_masses": [{"region": "weight", "mass": 1000.0}],
             "constraints": [{"region": "top", "dofs": [1, 1, 1]}, {"region": "weight", "dofs": [1, 1, 0]}],
             "loads": [{"region": "weight", "vector": [0, 0, -T], "amplitude": "hold"}],
             "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.05, "dt_safety": 0.02,
                        "damping": {"type": "none"}},
             "output": {"path": str(tmp_path / f"t{every}.h5"), "fields": ["displacement", "member_failed"],
                        "every": every}}
        box = {}
        run_case(d, on_ready=lambda s, p: box.setdefault("s", s))
        s = box["s"]
        assert type(s).__name__ == "TrussSolver" and len(s.breaking.failure_events) == 1
        ends.append((s.u.copy(), s.v.copy(), np.asarray(s.breaking.failure_events, float)))
    for u, v, ev in ends[1:]:
        assert np.array_equal(u, ends[0][0]) and np.array_equal(v, ends[0][1]) and np.array_equal(ev, ends[0][2])


@pytest.mark.parametrize("when", ["at the end", "in the last step"])
def test_191_a_removal_at_the_end_of_the_dynamic_step_is_there_for_the_relaxation_after_it(tmp_path, when):
    """[dynamic, relaxation]: the last frame of the dynamic step closes the
    dynamic history -- a removal due at its very end is found there, since
    no dynamic step follows. Asking for "the last step of all" dropped it,
    and the relaxation ran with the member still in (found in review)."""
    T = 0.02
    d = _hanging(tmp_path, "dr", strength_over_static=100.0, removal=1.0, total=T)
    box = {}
    prep_dt = {}
    try:
        run_case(dict(d, output=dict(d["output"], path=str(tmp_path / "dt.h5"))),
                 on_ready=lambda s, p: prep_dt.setdefault("dt", s.dt), on_progress=lambda *a: False, progress_every=1)
    except Exception:
        pass
    d["removals"] = [{"region": "cable", "time": T if when == "at the end" else T - 0.5 * prep_dt["dt"], "split": False}]
    d["steps"] = [{"name": "a", "procedure": "dynamic", "total_time": T},
                  {"name": "b", "procedure": "relaxation", "load_steps": 1, "max_steps_per_load": 2000}]
    for every in (1, 10 ** 6):
        d["output"]["every"] = every
        box.clear()
        try:
            run_case(d, on_ready=lambda s, p: box.setdefault("s", s))
        except Exception:
            pass                                    # the weight has nothing left to hang from
        s = box["s"]
        events = np.asarray(s.breaking.failure_events, float)
        assert len(events) == 1 and events[0, 0] == pytest.approx(T, rel=1e-9), (every, events)
        assert not s.breaking.member_active[int(events[0, 1])]
