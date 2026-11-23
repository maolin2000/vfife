"""The implicit static solver takes rigid diaphragms and member failure, and
an increment Newton cannot take falls back to relaxation (1.7).

* A rigid floor is a linear multi-point constraint: Newton solves the
  master system T^T K T. Checked on a one-storey frame under a lateral load
  at one corner: the floor's in-plane motion is one rigid motion (the same
  rotation at every node, the translations differing by rotation times
  lever arm), the vertical dofs are free, and the reactions balance the
  load.
* A member that reaches its capacity in an accepted equilibrium is taken
  out and the level solved again without it, and again while that
  cascades: a girder hung from two hangers under a growing tip load loses
  the weak one at the level where its force passes its strength, the other
  takes the load; with both too weak the girder ends hanging from its pin.
  A split-off or a timed removal has no static meaning and is refused by
  name.
* A shallow arch under a load past its limit point: Newton without a line
  search cannot take the snap, the relaxation fallback does, and Newton
  goes on from the inverted arch; `fallback: none` stops with the error.
"""
import os
import pathlib
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_member_failure import G, STEEL, _write_msh      # noqa: E402

from vfife.api import run_case                             # noqa: E402


def _run(d):
    box = {}
    out = run_case(d, on_ready=lambda s, p: box.setdefault("s", s))
    return out, box["s"]


# ----------------------------------------------------------------- diaphragm
def _storey(tmp_path, diaphragm=True, mode="static", normal="z"):
    """A one-storey frame; `normal: y` is the same building with its y and z
    axes swapped (the floor's normal along y)."""
    sw = (lambda v: [v[0], v[2], v[1]]) if normal == "y" else (lambda v: v)
    pts = {}
    for k, zz in enumerate((0.0, 3.0)):
        for j, (x, y) in enumerate([(0, 0), (4, 0), (4, 5), (0, 5)]):
            pts[f"N{k}{j}"] = sw([x, y, zz])
    lines = {f"c{j}": {"through": [f"N0{j}", f"N1{j}"], "size": 3.0} for j in range(4)}
    lines.update({f"b{j}": {"through": [f"N1{j}", f"N1{(j + 1) % 4}"], "size": 10.0} for j in range(4)})
    d = {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts, "lines": lines,
                  "groups": {"cols": [f"c{j}" for j in range(4)], "beams": [f"b{j}" for j in range(4)]}},
         "regions": {"base": {"plane": {"point": [0, 0, 0], "normal": sw([0, 0, 1])}},
                     "floor": {"plane": {"point": sw([0, 0, 3.0]), "normal": sw([0, 0, 1])}}},
         "materials": {"c": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2400.0}},
         "sections": {"s": {"type": "beam", "material": "c", "shape": "general", "area": 0.16,
                            "iy": 2.1e-3, "iz": 2.1e-3, "j": 3.6e-3}},
         "assign": [{"region": "cols", "section": "s", "orientation": [1, 0, 0]},
                    {"region": "beams", "section": "s", "orientation": sw([0, 0, 1])}],
         "constraints": [{"region": "base", "dofs": [1] * 6}],
         "loads": [{"region": "N10", "vector": sw([2.0e5, 0, -1.0e5])}],
         "solver": {"mode": mode, "backend": "numpy", "tolerance": 1e-8, "load_steps": 2},
         "output": {"path": str(tmp_path / f"storey_{int(diaphragm)}_{normal}.h5"), "fields": ["displacement"],
                    "every": 1,
                    "monitors": {"Rx": {"kind": "reaction", "region": "base", "dof": "x"},
                                 "Rz": {"kind": "reaction", "region": "base", "dof": normal}}}}
    if diaphragm:
        d["diaphragms"] = [{"region": "floor", "normal": normal}]
    return d


def test_a_rigid_floor_moves_as_one_in_its_plane_and_bends_out_of_it(tmp_path):
    o, s = _run(_storey(tmp_path))
    assert o.status == "CONVERGED"
    q = s.q.reshape(-1, 6)
    floor = s._region("floor")
    x = s.mesh.nodes[floor]
    rz = q[floor, 5]
    # the constraint is on the spins (the rotation increments); the logged
    # rotations agree to the second order of the bending rotations (3e-7)
    assert np.allclose(rz, rz[0], rtol=1e-5, atol=0)
    assert abs(rz[0]) > 1e-5                                    # the corner load twists the floor
    c = x[:, :2].mean(axis=0)
    ux = q[floor, 0]; uy = q[floor, 1]
    # u = U - alpha r_b, v = V + alpha r_a about any point
    assert np.allclose(ux - ux.mean(), -rz[0] * (x[:, 1] - c[1]), rtol=0, atol=1e-6 * abs(ux).max())
    assert np.allclose(uy - uy.mean(), rz[0] * (x[:, 0] - c[0]), rtol=0, atol=1e-6 * abs(ux).max())
    # out of its plane the floor is not rigid: the loaded corner sags most
    w = q[floor, 2]
    assert w.min() < 0.0 and np.ptp(w) > 0.5 * abs(w.min())
    # the reactions balance the load (the master residual vanished, and the
    # constraint forces are internal to the floor)
    assert np.isclose(float(np.asarray(o.monitors["Rx"])[-1]), -2.0e5, rtol=1e-7)
    assert np.isclose(float(np.asarray(o.monitors["Rz"])[-1]), 1.0e5, rtol=1e-7)


def test_a_floor_normal_to_y_turns_the_way_its_rotation_says(tmp_path):
    """The same building with y and z swapped: the floor's plan rotation is
    the one the z-normal building has, and the nodal rotation about the
    normal IS that rotation -- it was its negative before 1.7 (the (a, b)
    axes were not cyclic with the normal; found in review)."""
    oz, sz = _run(_storey(tmp_path, normal="z"))
    oy, sy = _run(_storey(tmp_path, normal="y"))
    qz = sz.q.reshape(-1, 6); qy = sy.q.reshape(-1, 6)
    fz = sz._region("floor"); fy = sy._region("floor")
    rz = qz[fz, 5]; ry = qy[fy, 4]
    assert np.allclose(ry, ry[0], rtol=1e-5, atol=0)
    assert abs(abs(ry[0]) - abs(rz[0])) < 1e-6 * abs(rz[0])        # congruent buildings
    # in the y-normal building the plan axes are (z, x): u_z = U_z - alpha r_x,
    # u_x = U_x + alpha r_z, with alpha the nodal rotation about y itself
    x = sy.mesh.nodes[fy]
    rx = x[:, 0] - x[:, 0].mean(); rzz = x[:, 2] - x[:, 2].mean()
    uz = qy[fy, 2]; ux = qy[fy, 0]
    assert np.allclose(uz - uz.mean(), -ry[0] * rx, rtol=0, atol=1e-6 * abs(ux).max())
    assert np.allclose(ux - ux.mean(), ry[0] * rzz, rtol=0, atol=1e-6 * abs(ux).max())
    assert np.isclose(abs(qy[fy, 1]).max(), abs(qz[fz, 2]).max(), rtol=1e-6)   # the sag out of plane


def test_without_the_floor_the_corner_drifts_on_its_own(tmp_path):
    o, s = _run(_storey(tmp_path, diaphragm=False))
    q = s.q.reshape(-1, 6)
    floor = s._region("floor")
    x = s.mesh.nodes[floor]
    rz = q[floor, 5]; ux = q[floor, 0]
    off = ux - ux.mean() + rz.mean() * (x[:, 1] - x[:, 1].mean())
    assert np.ptp(rz) > 1e-3 * abs(rz).max()                  # the beams bend in plan
    assert abs(off).max() > 1e-3 * abs(ux).max()               # and the plan is not rigid


# ------------------------------------------------------------ member failure
def _hung_girder(tmp_path, strength_right, strength_mid=1e12, mode="static", split=None, removal=False):
    _write_msh(tmp_path / "hg.msh", [(0, 0, 0), (2, 0, 0), (4, 0, 0), (2, 0, 3), (4, 0, 3)],
               {"pin": (0, [0]), "anchors": (0, [3, 4]), "tip": (0, [2]),
                "girder": (1, [(0, 1), (1, 2)]), "mid": (1, [(3, 1)]), "right": (1, [(4, 2)])})
    A = 1e-3
    fail_r = {"tensile_strength": strength_right / A}
    if split is not None:
        fail_r["split"] = split
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / "hg.msh")},
         "materials": {"steel": STEEL},
         "sections": {"g": {"type": "beam", "material": "steel", "shape": "rect", "width": 0.3, "height": 0.5},
                      "h": {"type": "bar", "material": "steel", "shape": "general", "area": A}},
         "assign": [{"region": "girder", "section": "g", "orientation": [0, 1, 0]},
                    {"region": "mid", "section": "h", "tension_only": True,
                     "failure": {"tensile_strength": strength_mid / A}},
                    {"region": "right", "section": "h", "tension_only": True, "failure": fail_r}],
         "constraints": [{"region": "anchors", "dofs": [1] * 6}, {"region": "pin", "dofs": [1, 1, 1, 1, 0, 1]}],
         "amplitudes": {"ramp": [[0, 0], [1, 1]]},
         "loads": [{"region": "tip", "vector": [0, 0, -1.0e5], "amplitude": "ramp"}],
         "solver": {"mode": mode, "backend": "numpy", "tolerance": 1e-6, "load_steps": 10},
         "output": {"path": str(tmp_path / f"hg_{mode}.h5"), "fields": ["displacement", "member_failed"], "every": 1,
                    "monitors": {"wt": {"kind": "displacement", "region": "tip", "dof": "z"}}}}
    if removal:
        d["removals"] = [{"region": "mid", "time": 0.5}]
    return d


def test_a_hanger_that_breaks_hands_its_load_to_the_other(tmp_path):
    # the right hanger carries P (the mid one is slack until it stretches):
    # at 0.6e5 of strength it breaks at the 0.8 level, and the mid one holds
    o, s = _run(_hung_girder(tmp_path, 0.6e5))
    assert o.status == "CONVERGED"
    assert list(np.flatnonzero(s.breaking.member_failed)) == [3]
    ev = np.asarray(s.breaking.failure_events)
    assert ev.shape == (1, 4) and ev[0, 1] == 3 and ev[0, 3] == 0.0
    assert 0.79 < ev[0, 0] < 0.81 and 1.0 <= ev[0, 2] < 1.2
    wt = np.asarray(o.monitors["wt"])
    # the record is of equilibria: the level at which it broke is recorded
    # with the load on the mid hanger -- a jump in the tip deflection
    assert wt[6] > -1e-3 and wt[7] < -4e-3 and wt[-1] < wt[7]
    assert s.diagnostics["member_failures"] == s.breaking.failure_events
    with h5py.File(tmp_path / "hg_static.h5") as f:
        last = sorted(f["steps"])[-1]
        mf = f["steps"][last]["member_failed"][...]
    assert mf[3] == 1.0 and mf[2] == 0.0


def test_the_cascade_leaves_the_girder_hanging_from_its_pin(tmp_path):
    o, s = _run(_hung_girder(tmp_path, 0.6e5, strength_mid=1.0e5))
    assert o.status == "CONVERGED"
    assert sorted(np.flatnonzero(s.breaking.member_failed).tolist()) == [2, 3]
    ev = np.asarray(s.breaking.failure_events)
    assert ev.shape == (2, 4) and abs(ev[0, 0] - ev[1, 0]) < 1e-12     # the same level, two passes
    assert float(np.asarray(o.monitors["wt"])[-1]) < -3.9                # hanging vertically: 4 m down


def test_nothing_breaks_below_its_strength(tmp_path):
    o, s = _run(_hung_girder(tmp_path, 2.0e5))
    assert o.status == "CONVERGED" and not s.breaking.member_failed.any()


def test_what_a_static_solution_cannot_break_is_refused_by_name(tmp_path):
    from vfife.core.diagnostics import UserError
    with pytest.raises((ValueError, UserError), match="split: true"):
        _run(_hung_girder(tmp_path, 0.6e5, split=True))
    with pytest.raises((ValueError, UserError), match="removals at a time"):
        _run(_hung_girder(tmp_path, 0.6e5, removal=True))


# ------------------------------------------------------------------ fallback
def _arch(tmp_path, fallback, P=2.0e5):
    return {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "C": [1.0, 0, 0.1], "B": [2.0, 0, 0]},
                     "lines": {"arch": {"through": ["A", "C", "B"], "size": 0.25}}},
            "materials": {"s": {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}},
            "sections": {"r": {"type": "beam", "material": "s", "shape": "rect", "width": 0.02, "height": 0.02}},
            "assign": [{"region": "arch", "section": "r", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "A", "dofs": [1, 1, 1, 1, 0, 1]}, {"region": "B", "dofs": [1, 1, 1, 1, 0, 1]},
                            {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
            "amplitudes": {"ramp": [[0, 0], [1, 1]]},
            "loads": [{"region": "C", "vector": [0, 0, -P], "amplitude": "ramp"}],
            # no line search: Newton cannot take the snap; three cutbacks
            # before it gives up
            "solver": {"mode": "static", "backend": "numpy", "tolerance": 1e-6, "load_steps": 10,
                       "max_steps_per_load": 200000,
                       "static": {"fallback": fallback, "line_search": False, "cutbacks": 3}},
            "output": {"path": str(tmp_path / f"arch_{fallback}.h5"), "fields": ["displacement"], "every": 1,
                       "monitors": {"w": {"kind": "displacement", "region": "C", "dof": "z"}}}}


def test_an_increment_newton_cannot_take_is_relaxed_and_newton_goes_on(tmp_path, capsys):
    o, s = _run(_arch(tmp_path, "relaxation"))
    assert o.status == "CONVERGED"
    w = np.asarray(o.monitors["w"])
    assert len(w) == 10 and w[0] > -0.02                       # the arch, before the snap
    assert w[1] < -0.2 and w[-1] < w[1]                         # inverted, and loaded on
    assert s._newton.fallbacks == 10 - 1                       # the rest of the step relaxed
    log = capsys.readouterr().out
    assert "fallback" in log and "relaxed in" in log
    # Newton's own record: the residuals are the increments', one per level,
    # and every relaxed level is polished under the tolerance
    assert len(s.residuals) == 10 and max(s.residuals) <= 1e-6


def test_without_the_fallback_the_limit_point_is_an_error(tmp_path):
    o, s = _run(_arch(tmp_path, "none"))
    assert o.status == "MAX_STEPS" and "limit point" in (o.message or "")
    assert len(np.asarray(o.monitors["w"])) == 1


@pytest.mark.parametrize("normal", ["z", "y"])
def test_a_relaxation_with_a_rigid_floor_converges_to_the_static_answer(tmp_path, normal):
    """`mode: relaxation` measured its residual on the nodes, where a
    floor's constraint forces never vanish: the storey sat at 0.5 after
    200 000 iterations (found in the review of 1.7). It measures the
    floors' own equations now (1.9)."""
    o1, s1 = _run(_storey(tmp_path, normal=normal))
    d = _storey(tmp_path, normal=normal, mode="relaxation")
    d["solver"].update({"n_steps": 400000, "max_steps_per_load": 200000, "tolerance": 1e-7,
                        "kinetic_tolerance": 1e-14})
    d["output"]["path"] = str(tmp_path / f"rel_{normal}.h5")
    o2, s2 = _run(d)
    assert o2.status == "CONVERGED"
    assert np.abs(s2.q - s1.q).max() < 1e-5 * np.abs(s1.q).max()


def test_the_relaxation_fallback_takes_a_rigid_floor(tmp_path):
    """Refused in 1.7 (the relaxation's residual was the nodal one); a
    Newton held to one iteration and no cutback hands both levels over."""
    o1, s1 = _run(_storey(tmp_path))
    d = _storey(tmp_path)
    d["solver"].update({"max_steps_per_load": 200000,
                        "static": {"fallback": "relaxation", "max_iterations": 1, "cutbacks": 0}})
    d["output"]["path"] = str(tmp_path / "fb.h5")
    o2, s2 = _run(d)
    assert o2.status == "CONVERGED"
    assert s2._newton.fallbacks >= 1
    assert np.abs(s2.q - s1.q).max() < 1e-5 * np.abs(s1.q).max()


def test_a_long_fallback_relaxation_can_be_stopped(tmp_path):
    """The fallback reports every 500 iterations, as a staged relaxation
    does, and stops when asked (1.9): it ran its whole
    budget deaf before."""
    from vfife.core.diagnostics import AnalysisInterrupted
    seen = []

    def stop(level, n, resid, ke, inc):
        seen.append(ke)
        return not ke > 0.0          # Newton reports no kinetic energy; the relaxation does

    with pytest.raises(AnalysisInterrupted):
        run_case(_arch(tmp_path, "relaxation"), on_progress=stop)
    assert seen[-1] > 0.0


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_stop_in_the_middle_of_a_fallback_resumes_from_the_last_increment(tmp_path, backend):
    """The fallback's reports reach the progress callback, where a stop is
    taken and a checkpoint written. Mid-relaxation the solver holds a
    half-relaxed state, the fictitious mass and the relaxation's iteration
    count: the checkpoint resumed Newton from there (found in the review of
    1.9). No checkpoint is written inside a fallback, a
    stop puts the last increment back, and the restart ends where the
    uninterrupted run ends."""
    if backend == "numba":
        pytest.importorskip("numba")
    from vfife.core.diagnostics import AnalysisInterrupted

    def case(name):
        d = _arch(tmp_path, "relaxation"); d["solver"]["backend"] = backend
        d["output"]["path"] = str(tmp_path / f"{name}.h5")
        return d
    _ow, sw = _run(case("whole"))
    n = {"k": 0}

    def stop(level, nn, resid, ke, inc):
        if ke > 0.0:                         # a fallback's own report
            n["k"] += 1
            if n["k"] == 3:
                return False
        return None
    with pytest.raises(AnalysisInterrupted):
        run_case(case("part"), on_progress=stop)
    box = {}
    rest = run_case(case("rest"), restart=str(tmp_path / "part.h5"), on_ready=lambda s, p: box.setdefault("s", s))
    assert rest.status == "CONVERGED"
    assert np.array_equal(box["s"].q, sw.q)
    assert np.array_equal(box["s"].mass, sw.mass)
