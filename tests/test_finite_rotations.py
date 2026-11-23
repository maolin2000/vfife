"""Finite rotations of frame members in space.

THE DEFECT THIS EXISTS FOR
    A member's rigid rotation used to be removed by subtracting rotation
    vectors, with the element frame following the chord alone. Exact in a
    plane; in space a rigidly rotated-and-twisted member carried a spurious
    force, and the 45-degree bend (case_18) came out 2% stiff in the load
    direction and 5% off across the chord. The tests below pin the exact
    kinematics: a rigid motion is force-free, the planar case reduces to
    the old formula, the batched frames equal an independent scipy version,
    and the bend lands on Bathe & Bolourchi.
"""
import pathlib
import tempfile

import numpy as np
import pytest

from benchmarks.generate_bend45 import build as build_bend
from vfife.api import prepare_case, run_case
from vfife.cli import _with_element, make_solver
from vfife.core.rotations import beam_frames, exp_map, log_map, orthonormalize


def _bend_problem(n=8, load=600.0, levels=12, tol=1e-6, j=1.0 / 6.0):
    d = pathlib.Path(tempfile.mkdtemp())
    build_bend(d / "bend.msh", n)
    return {
        "name": "bend45_test", "units": {"length": "in", "force": "lbf", "stress": "psi"},
        "mesh": {"type": "gmsh", "path": str(d / "bend.msh")},
        "materials": {"mat": {"type": "linear_elastic", "E": 1.0e7, "nu": 0.0, "rho": 2.5e-4}},
        "sections": {"square": {"type": "beam", "material": "mat", "shape": "general",
                                "area": 1.0, "iy": 1.0 / 12.0, "iz": 1.0 / 12.0, "j": j}},
        "assign": [{"region": "members", "section": "square", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
        "steps": [{"name": "load", "loads": [{"region": "tip", "vector": [0, 0, load]}],
                   "load_steps": levels, "max_steps_per_load": 100000, "tolerance": tol}],
        "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 1200000,
                   "damping": {"type": "kinetic"}, "tolerance": tol},
        "output": {"path": str(d / "r.h5"), "every": 10**7,
                   "monitors": {"u": {"kind": "displacement", "region": "tip", "dof": "x"},
                                "v": {"kind": "displacement", "region": "tip", "dof": "y"},
                                "w": {"kind": "displacement", "region": "tip", "dof": "z"}}},
    }


def _solver(problem):
    pr = prepare_case(problem)
    return make_solver(_with_element(pr.case, pr.element, pr.case_path), pr.mesh, pr.material)


# ---- the maps ----------------------------------------------------------- #

def test_exp_and_log_are_inverse_below_pi():
    rng = np.random.default_rng(0)
    th = rng.normal(size=(200, 3))
    th *= (rng.uniform(1e-9, np.pi - 1e-3, size=200) / np.linalg.norm(th, axis=1))[:, None]
    assert np.allclose(log_map(exp_map(th)), th, rtol=1e-9, atol=1e-12)
    R = exp_map(th)
    assert np.allclose(np.einsum("eij,ekj->eik", R, R), np.eye(3), atol=1e-13)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-13)


def test_a_planar_rotation_reduces_to_the_old_difference():
    """Member along x, everything about z: theta_loc = theta_node - phi."""
    phi, t1, t2 = 0.7, 0.9, 0.5
    x = np.array([[[0, 0, 0], [np.cos(phi), np.sin(phi), 0]]], float)
    theta = np.array([[[0, 0, t1], [0, 0, t2]]], float)
    R, loc, L = beam_frames(x, theta, np.eye(3)[None])
    assert np.allclose(loc[0, :, 2], [t1 - phi, t2 - phi], atol=1e-14)
    assert np.abs(loc[0, :, :2]).max() < 1e-14


def test_the_frames_are_objective():
    """A superposed rigid rotation Q leaves the local rotations unchanged
    and rotates the frame by Q."""
    rng = np.random.default_rng(2)
    R0 = np.linalg.qr(rng.normal(size=(3, 3)))[0]; R0 *= np.sign(np.linalg.det(R0)); R0 = R0[None]
    x = rng.normal(size=(1, 2, 3)); theta = rng.normal(size=(1, 2, 3)) * 0.4
    Q = exp_map(np.array([0.4, -1.1, 0.8]))
    Ra, la, _ = beam_frames(x, theta, R0)
    Rb, lb, _ = beam_frames(np.einsum("ij,enj->eni", Q, x), log_map(Q[None, None] @ exp_map(theta)), R0)
    assert np.allclose(la, lb, atol=1e-13)
    assert np.allclose(Rb, np.einsum("eij,kj->eik", Ra, Q), atol=1e-13)


# ---- the solver --------------------------------------------------------- #

def test_a_rigidly_moved_and_twisted_frame_carries_no_force():
    """The whole bend rotated by a large rotation Q about a skew axis and
    translated: every node's rotation vector is log Q, and the internal
    force must vanish. Under the old chord-only frame the twist about each
    member's own axis was invisible and this produced moments."""
    s = _solver(_bend_problem(8))
    Q = exp_map(np.array([0.9, -0.6, 1.3]))
    X = s.mesh.nodes
    x = (Q @ X.T).T + np.array([3.0, -2.0, 5.0])
    q = np.zeros((s.nn, 6)); q[:, :3] = x - X; q[:, 3:] = log_map(Q)
    s.q = q.ravel(); s.orient = np.broadcast_to(Q, (s.nn, 3, 3)).copy()
    f, _ = s.internal_force()
    scale = float(np.abs(np.asarray(s.k_local)).max())
    assert np.abs(f).max() < 1e-10 * scale


def test_batched_frame_forces_equal_the_scipy_reference_in_space():
    s = _solver(_bend_problem(8))
    rng = np.random.default_rng(3)
    s.q = rng.normal(size=s.ndof) * 0.3
    s.orient = exp_map(s.dofs.view(s.q)[:, 3:])
    f_b, a_b = s.internal_force()
    f_r, a_r = s._internal_force_reference()
    assert np.max(np.abs(f_b - f_r)) < 1e-12 * np.max(np.abs(f_r))
    assert np.max(np.abs(a_b - a_r)) < 1e-12 * np.max(np.abs(a_r))


def test_the_45_degree_bend_lands_on_bathe_and_bolourchi():
    """Four elements, P = 600: (-13.4, -23.5, 53.4) is the 1979 answer with
    eight. Four give (-13.31, -23.54, 53.68); the old kinematics gave
    (-12.8, -24.2, 52.4) with eight and would fail every bound here."""
    r = run_case(_bend_problem(4, 600.0, 6, 1e-5))
    assert r.converged
    u, v, w = (r.monitors[k][-1] for k in ("u", "v", "w"))
    assert abs(u / -13.4 - 1.0) < 0.02
    assert abs(v / -23.5 - 1.0) < 0.01
    assert abs(w / 53.4 - 1.0) < 0.01


# ---- the orientation state ------------------------------- #

def test_the_orientation_is_advanced_multiplicatively_and_stays_a_rotation():
    """Spins about alternating axes do not commute: after them the
    orientation is the ordered product of the exponentials, not the
    exponential of the sum, and it is orthogonal to round-off."""
    s = _solver(_bend_problem(4))
    spins = [np.array([0.3, 0.0, 0.0]), np.array([0.0, 0.4, 0.0])] * 50
    expect = np.eye(3)
    for w in spins:
        s.dofs.view(s.v)[:, 3:] = w
        s._advance(1.0)
        expect = exp_map(w) @ expect
    assert np.allclose(s.orient[0], expect, atol=1e-12)
    assert not np.allclose(s.orient[0], exp_map(np.sum(spins, axis=0)), atol=1e-3)
    assert np.allclose(s.orient[0] @ s.orient[0].T, np.eye(3), atol=1e-13)
    assert np.allclose(s.dofs.view(s.q)[0, 3:], log_map(expect), atol=1e-12)


def test_a_symmetry_plane_holds_its_node_in_the_subgroup_exactly():
    """Two rotation components held (a symmetry plane): the spin has one
    free component, and every product of rotations about one axis is a
    rotation about that axis, so the held logarithm components stay at
    zero exactly -- not at second order."""
    s = _solver(_bend_problem(4))
    fixed = np.zeros(s.ndof, bool)
    fx = s.dofs.view(fixed); fx[:, 3] = True; fx[:, 5] = True         # r_x, r_z held everywhere
    rng = np.random.default_rng(4)
    for _ in range(200):
        s.dofs.view(s.v)[:, 3:] = rng.normal(size=(s.nn, 3))
        s.v[fixed] = 0.0
        s._advance(0.05)
        s._apply_bc(fixed, np.zeros(s.ndof, bool), [], 1.0)
    qr = s.dofs.view(s.q)[:, 3:]
    assert np.abs(qr[:, [0, 2]]).max() < 1e-14
    assert np.abs(qr[:, 1]).max() > 0.05                  # it did move


def test_a_prescribed_rotation_rebuilds_the_orientation():
    s = _solver(_bend_problem(4))
    node = 2
    dofs = np.array([s.dofs.node_dofs(node)[0, 4]])          # r_y of one node
    fixed = np.zeros(s.ndof, bool); fixed[dofs] = True
    a = lambda p: p
    s._apply_bc(fixed, np.zeros(s.ndof, bool), [(dofs, 0.8, a, "displacement")], 1.0)
    assert np.allclose(s.orient[node], exp_map(np.array([0.0, 0.8, 0.0])), atol=1e-14)
    assert np.allclose(s.dofs.view(s.q)[node, 3:], [0.0, 0.8, 0.0])


def test_a_checkpoint_carries_the_orientation_and_an_old_one_rebuilds_it():
    s = _solver(_bend_problem(4))
    s.dofs.view(s.v)[:, 3:] = [0.2, -0.1, 0.3]; s._advance(1.0)
    ck = s.checkpoint()
    t = _solver(_bend_problem(4)); t.restore(ck)
    assert np.allclose(t.orient, s.orient, atol=0.0)
    old = {k: v for k, v in ck.items() if k != "orient"}
    u = _solver(_bend_problem(4)); u.restore(old)
    assert np.allclose(u.orient, s.orient, atol=1e-13)
