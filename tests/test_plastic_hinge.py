"""Plastic hinges at the ends of mixed_line beams.

A beam end whose |N|/N_p + |M_y|/M_y,p + |M_z|/M_z,p reaches 1 turns plastic:
its moment stays on the surface while the end rotates, the rotation is
dissipated, and the end breaks only when the accumulated plastic rotation
reaches `rotation_capacity`. The closed forms: a cantilever's root moment
cannot pass M_p = f_y I / c whatever the tip load, below it nothing changes,
and the energy the hinge takes is booked (the ledger closes).
"""
import h5py
import numpy as np
import pytest

from test_member_failure import STEEL, _run, _write_msh

FY, C, I, A = 3.0e8, 0.1, 1e-5, 1e-2
M_P = FY * I / C                                 # 30 kN m (shape factor 1)
L = 2.0


def _cantilever(tmp_path, name, amp, total, theta_u=0.05, P=M_P / L, damping=50.0, backend="numpy"):
    n = 4
    xs = np.linspace(0, L, n + 1)
    _write_msh(tmp_path / f"{name}.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [n]),
                "first": (1, [(0, 1)]), "rest": (1, [(k, k + 1) for k in range(1, n)])})
    beam = {"type": "beam", "material": "steel", "shape": "general", "area": A, "iy": I, "iz": I, "j": 2 * I}
    fail = {"yield_strength": FY, "extreme_fibre": [C, C], "rotation_capacity": theta_u}
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": f"{name}.msh"},
        "materials": {"steel": STEEL},
        "sections": {"b": beam},
        "assign": [{"region": "first", "section": "b", "orientation": [0, 0, 1], "failure": fail},
                   {"region": "rest", "section": "b", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
        "amplitudes": {"a": amp},
        "loads": [{"region": "tip", "vector": [0, 0, -P], "amplitude": "a"}],
        "solver": {"backend": backend, "mode": "dynamic", "total_time": total,
                   "damping": {"type": "viscous", "coeff": damping}},
        "output": {"fields": ["displacement", "bending_y", "bending_z", "capacity_ratio", "hinge_rotation", "hinge_usage"],
                   "every": 100, "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}},
    }
    out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        steps = sorted(f["steps"])
        T = np.array([f["steps"][k].attrs["time"] for k in steps])
        M = np.array([max(abs(f["steps"][k]["bending_y"][0]), abs(f["steps"][k]["bending_z"][0])) for k in steps])
        # the usage (rotation / capacity); `hinge_rotation` is the rotation in
        # radians since v1.0
        H = np.array([f["steps"][k]["hinge_usage"][0] for k in steps])
        ev = np.asarray(f["diagnostics/member_failures"]) if "member_failures" in f["diagnostics"] else np.zeros((0, 4))
        err = np.asarray(f["diagnostics/relative_energy_error"])
    return out, T, M, H, ev, err


def test_the_root_moment_stops_at_the_plastic_moment_and_breaks_on_rotation(tmp_path):
    # tip load 0 -> 1.5 P_p over 3 s (an amplitude's abscissa is the run's
    # progress, 0..1): past P_p = M_p / L the root is a hinge,
    # the cantilever swings down about it, and the root tears only when it
    # has turned through the rotation capacity -- NOT when M reached M_p
    out, T, M, H, ev, err = _cantilever(tmp_path, "hinge", [[0.0, 0.0], [1.0, 1.5]], 3.0)
    t_yield = 3.0 / 1.5                                   # P = P_p
    assert len(ev) >= 1 and int(ev[0, 1]) == 0
    t_break = ev[0, 0]
    assert t_break > t_yield + 0.05                        # the brittle root went at t_yield
    before = T < t_break - 1e-9
    assert M[before].max() == pytest.approx(M_P, rel=0.02)            # capped
    late = before & (T > t_yield + 0.02)
    assert M[late].min() > 0.97 * M_P                                  # and held there
    assert np.all(np.isnan(H) | (H <= 1.001))                            # one step past it at most
    assert H[before][-1] > 0.8                                         # the rotation built up to it
    assert H[T < t_yield - 0.05].max() == pytest.approx(0.0, abs=1e-9) # nothing before yield
    assert np.abs(err).max() < 2e-2                                    # the hinge's work is booked


def test_the_numba_kernel_sees_the_same_hinge(tmp_path):
    # the plastic rotations ride in the eigen-deformation the kernel already
    # takes (a temperature's slot): the same break, the same moments
    pytest.importorskip("numba")
    a = _cantilever(tmp_path, "np", [[0.0, 0.0], [1.0, 1.5]], 2.5)
    b = _cantilever(tmp_path, "nb", [[0.0, 0.0], [1.0, 1.5]], 2.5, backend="numba")
    assert b[4][0, 0] == pytest.approx(a[4][0, 0], rel=1e-3)
    assert np.allclose(b[2], a[2], rtol=1e-4, atol=1e-6 * M_P)


def test_below_the_plastic_moment_nothing_is_plastic(tmp_path):
    out, T, M, H, ev, err = _cantilever(tmp_path, "elastic", [[0.0, 0.0], [0.25, 0.8], [1.0, 0.8]], 2.0)
    assert len(ev) == 0
    assert np.nanmax(H) == 0.0
    assert M[-1] == pytest.approx(0.8 * M_P, rel=0.02)


def test_rotation_capacity_is_refused_where_it_would_be_ignored():
    from vfife.io.schema import FailureSpec
    with pytest.raises(ValueError, match="beam's"):
        FailureSpec(tensile_strength=1e9, rotation_capacity=0.05)
    assert FailureSpec(yield_strength=3e8, extreme_fibre=[0.1, 0.1], rotation_capacity=0.05).rotation_capacity == 0.05
