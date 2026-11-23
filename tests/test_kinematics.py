"""Element kinematics: the tests that catch co-rotational and B-matrix bugs.

test_rigid_body_rotation is the single most valuable test for this formulation
and is deliberately the first one in the file.  A rigid rotation must produce
exactly zero internal force; anything wrong in the polar decomposition or in
the hourglass base vectors shows up here immediately and nowhere else.
"""

import numpy as np
import pytest

from vfife.core.mesh import box_mesh, scatter_add
from vfife.elements import continuum
from vfife.materials import create
from vfife.voigt import elastic_matrix

FORMULATIONS = ["full", "reduced", "b-bar"]


def _setup(formulation, nx=3, ny=2, nz=2, distort=0.0, seed=0):
    mesh = box_mesh((3.0, 2.0, 2.0), (nx, ny, nz))
    if distort:
        rng = np.random.default_rng(seed)
        inner = np.all(
            (mesh.nodes > 1e-9) & (mesh.nodes < np.array([3.0, 2.0, 2.0]) - 1e-9),
            axis=1,
        )
        mesh.nodes[inner] += rng.uniform(-distort, distort, (inner.sum(), 3))
    mat = create("linear_elastic", E=210e9, nu=0.3, rho=7800.0)
    pre = continuum.precompute(
        mesh.nodes, mesh.elements, formulation=formulation,
        hourglass_coeff=0.1, material=mat,
    )
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    return mesh, mat, pre, x0_e


def _forces(mesh, mat, pre, x0_e, u, rot_iters=60):
    state = mat.init_state(mesh.n_elem * pre.n_gp)
    quat = np.zeros((mesh.n_elem, 4))
    quat[:, 0] = 1.0
    f_e, sig = continuum.internal_force(
        pre, x0_e, u[mesh.elements], quat, mat, state, 1e-6,
        rot_iters=rot_iters, want_stress=True,
    )
    return scatter_add(f_e, mesh.elements, mesh.n_nodes), sig


@pytest.mark.parametrize("formulation", FORMULATIONS)
@pytest.mark.parametrize("distort", [0.0, 0.15])
def test_rigid_body_translation(formulation, distort):
    mesh, mat, pre, x0_e = _setup(formulation, distort=distort)
    u = np.tile(np.array([0.37, -1.2, 4.5]), (mesh.n_nodes, 1))
    F, sig = _forces(mesh, mat, pre, x0_e, u)
    assert np.abs(sig).max() < 1e-3
    assert np.abs(F).max() < 1e-3


@pytest.mark.parametrize("formulation", FORMULATIONS)
@pytest.mark.parametrize("distort", [0.0, 0.15])
def test_rigid_body_rotation(formulation, distort):
    """30 degrees about a tilted axis, plus a translation. Force must vanish."""
    mesh, mat, pre, x0_e = _setup(formulation, distort=distort)
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    th = np.deg2rad(30.0)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    u = mesh.nodes @ R.T - mesh.nodes + np.array([0.1, -0.2, 0.3])

    F, sig = _forces(mesh, mat, pre, x0_e, u)
    scale = mat.E * np.abs(mesh.nodes).max()
    assert np.abs(sig).max() / scale < 1e-9, "spurious stress under rigid rotation"
    assert np.abs(F).max() / (scale * 1.0) < 1e-9


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_patch_test(formulation):
    """A linear displacement field must give the exact constant stress."""
    mesh, mat, pre, x0_e = _setup(formulation)
    A = np.array([[1.0e-4, 3.0e-5, -2.0e-5],
                  [3.0e-5, -5.0e-5, 1.5e-5],
                  [-2.0e-5, 1.5e-5, 2.0e-5]])          # symmetric -> pure strain
    u = mesh.nodes @ A.T
    _, sig = _forces(mesh, mat, pre, x0_e, u)

    eps = np.array([A[0, 0], A[1, 1], A[2, 2],
                    2 * A[0, 1], 2 * A[1, 2], 2 * A[2, 0]])
    exact = elastic_matrix(mat.E, mat.nu) @ eps
    err = np.abs(sig.reshape(-1, 6) - exact).max() / np.abs(exact).max()
    assert err < 1e-9, f"patch test failed for {formulation}: rel err {err:.2e}"


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_patch_test_equilibrium(formulation):
    """Under a constant stress field the interior nodes must be in equilibrium."""
    mesh, mat, pre, x0_e = _setup(formulation)
    A = np.eye(3) * 1e-4
    u = mesh.nodes @ A.T
    F, _ = _forces(mesh, mat, pre, x0_e, u)
    interior = np.all(
        (mesh.nodes > 1e-9) & (mesh.nodes < np.array([3.0, 2.0, 2.0]) - 1e-9), axis=1
    )
    if interior.any():
        scale = np.abs(F).max()
        assert np.abs(F[interior]).max() / scale < 1e-9


def test_warm_started_rotation_matches_svd():
    """The quaternion iteration must agree with the SVD polar decomposition."""
    rng = np.random.default_rng(3)
    n = 200
    A = rng.normal(size=(n, 3, 3))
    A += 3.0 * np.eye(3)                              # keep them near-rotations
    q = np.zeros((n, 4))
    q[:, 0] = 1.0
    continuum.extract_rotation(A, q, max_iter=100)
    R = continuum.quat_to_mat(q)

    U, _, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(U @ Vt))
    S = np.zeros((n, 3, 3))
    S[:, 0, 0] = S[:, 1, 1] = 1.0
    S[:, 2, 2] = d
    R_svd = U @ S @ Vt
    assert np.abs(R - R_svd).max() < 1e-12


def test_warm_start_converges_in_two_iterations():
    """A small incremental rotation must be tracked with 2 iterations."""
    rng = np.random.default_rng(5)
    n = 64
    A = rng.normal(size=(n, 3, 3)) * 0.05 + 2.0 * np.eye(3)
    q = np.zeros((n, 4))
    q[:, 0] = 1.0
    continuum.extract_rotation(A, q, max_iter=100)          # cold
    exact = continuum.quat_to_mat(q.copy())

    dth = 1e-4                                          # a typical per-step rotation
    K = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    dR = np.eye(3) + dth * K
    A2 = dR @ A
    continuum.extract_rotation(A2, q, max_iter=2)           # warm, only 2 iterations
    R2 = continuum.quat_to_mat(q)
    # residual must be a small fraction of the increment it had to track
    assert np.abs(R2 - dR @ exact).max() < 0.01 * dth


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_volumetric_locking_spurious_pressure(formulation):
    """The locking mechanism, measured directly at the element level.

    Under the bending-like field u_x = c*x*z a trilinear hexahedron produces a
    volumetric strain eps_v = c*z at its Gauss points, even though the exact
    incompressible response has eps_v = 0.  As nu -> 1/2 the bulk modulus
    multiplies that spurious dilatation into an enormous pressure, and the
    element locks.  b-bar replaces eps_v by its element average (zero here) and
    reduced integration only samples the centroid, so both must stay clean.

    Doing this at the element level avoids a confound found the hard way: at
    nu = 0.4999 the dynamic-relaxation solver converges extremely slowly
    (the Underwood masses are set by the bulk terms while the bending modes
    stay soft), so a solver-level locking test measures convergence, not
    locking.
    """
    nu = 0.4999
    # ONE element, so its centroid sits on the bending neutral plane and the
    # element-averaged dilatation is exactly zero
    mesh = box_mesh((1.0, 1.0, 1.0), (1, 1, 1))
    mat = create("linear_elastic", E=210e9, nu=nu, rho=7800.0)
    pre = continuum.precompute(mesh.nodes, mesh.elements, formulation=formulation,
                           hourglass_coeff=0.05, material=mat)
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))

    c = 1e-4
    z = mesh.nodes[:, 2] - 0.5
    u = np.zeros_like(mesh.nodes)
    u[:, 0] = c * (mesh.nodes[:, 0] - 0.5) * z

    _, sig = _forces(mesh, mat, pre, x0_e, u)
    pressure = np.abs(sig[..., :3].sum(axis=-1) / 3.0).max()

    K = mat.E / (3 * (1 - 2 * nu))
    ref = K * c * 0.5 / np.sqrt(3.0)        # bulk modulus x dilatation at a Gauss point
    if formulation == "full":
        assert pressure / ref > 0.9, (
            f"full integration produced no spurious pressure ({pressure/ref:.3e})"
        )
    else:
        assert pressure / ref < 1e-9, (
            f"{formulation} locks: spurious pressure {pressure/ref:.3e} of the "
            "full-integration scale"
        )
