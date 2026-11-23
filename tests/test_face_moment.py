"""A moment on a solid face: mesh.FaceMoment and the
`mass: eigenvalue` relaxation mass.

WHAT THIS PINS
    The face moment is a potential load, -|M| theta(x) with theta the angle
    the face's best-fit rotation has turned about M's axis: its nodal
    forces are minus that gradient (checked by finite differences on a
    warped face), have zero resultant and the couple M in the reference
    configuration, are the reference distribution rigidly rotated when the
    face turns about M's own axis -- through 180 degrees and beyond -- and
    are smooth through a full turn about a tilted axis. The first version
    re-evaluated the textbook traction on the current face; its tangent was
    almost purely skew and stalled the rolled cantilever (case_20) at
    m = 0.65 with a kinetic-damping thrash, and let an out-of-plane flutter
    grow from round-off at m = 0.4. The second, -M . phi of the rotation
    vector, was singular at a full turn.
"""
import numpy as np
import pytest

from vfife.api import prepare_case, run_case
from vfife.core.mesh import FaceMoment
from vfife.core.rotations import exp_map


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)


def _bar(divisions=(8, 2, 4), E=1.92e6, moment=(0.0, 1.0, 0.0), **solver):
    return {
        "name": "face_moment_test", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "box", "dimensions": [1.0, 0.05, 0.05], "divisions": list(divisions)},
        "materials": {"m": {"type": "linear_elastic", "E": E, "nu": 0.0, "rho": 1000.0}},
        "sections": {"body": {"type": "solid", "material": "m", "formulation": "reduced"}},
        "regions": {"root": {"plane": {"point": [0, 0, 0], "normal": [1, 0, 0]}},
                    "tip": {"plane": {"point": [1, 0, 0], "normal": [1, 0, 0]}}},
        "constraints": [{"region": "root", "dofs": [1, 1, 1]}],
        "loads": [{"region": "tip", "moment": list(moment), "follower": True}],
        "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 200000, "tolerance": 1e-6,
                   "damping": {"type": "kinetic"}, **solver},
        "output": {"path": "results/_face_moment_test.h5", "every": 10**7,
                   "monitors": {"u": {"kind": "displacement", "region": "tip", "dof": "x"},
                                "w": {"kind": "displacement", "region": "tip", "dof": "z"}}},
    }


@pytest.fixture(scope="module")
def face():
    pr = prepare_case(_bar())
    mesh = pr.mesh
    M = np.array([0.3, 6.283185307179586, -1.1])
    fm = FaceMoment(mesh.nodes, mesh.elements, "hexa8", mesh.regions["tip"], M)
    return mesh, fm, M


def _couple(fm, x, f):
    c = (fm.area[:, None] * x[fm.idx]).sum(axis=0) / fm.area.sum()
    return np.cross(x - c, f).sum(axis=0)


def test_the_reference_distribution_is_the_textbook_bending_traction(face):
    mesh, fm, M = face
    f = fm(mesh.nodes)
    assert np.allclose(f, fm.reference_forces())
    assert np.abs(f.sum(axis=0)).max() < 1e-12 * np.abs(f).max()
    assert np.allclose(_couple(fm, mesh.nodes, f), M)
    # the bending part is M z / I (tension on +z for a +y couple) times the
    # tributary area; the normal part (-1.1 about x) is an in-plane shear
    fm_y = FaceMoment(mesh.nodes, mesh.elements, "hexa8", mesh.regions["tip"], [0.0, M[1], 0.0])
    fy = fm_y(mesh.nodes)
    z = mesh.nodes[fm_y.idx, 2] - 0.025
    I = np.sum(fm_y.area * z**2)
    assert np.allclose(fy[fm_y.idx, 0], fm_y.area * M[1] * z / I)
    assert np.abs(fy[:, 1:]).max() < 1e-14 * np.abs(fy).max()
    fm_x = FaceMoment(mesh.nodes, mesh.elements, "hexa8", mesh.regions["tip"], [M[0], 0.0, 0.0])
    fx = fm_x(mesh.nodes)
    assert np.abs(fx[:, 0]).max() < 1e-14 * np.abs(fx).max() and np.abs(fx[:, 1:]).max() > 0.0
    assert np.allclose(_couple(fm_x, mesh.nodes, fx), [M[0], 0.0, 0.0])


@pytest.mark.parametrize("angle", [0.5, 2.6, np.pi - 1e-3, 3.3, 6.0])
def test_a_couple_about_its_own_axis_turns_with_the_face(face, angle):
    """Rotating the whole body about y: the y-couple is the reference
    distribution rotated, on both branches of the log map."""
    mesh, fm, _ = face
    fm_y = FaceMoment(mesh.nodes, mesh.elements, "hexa8", mesh.regions["tip"], [0.0, 2.0, 0.0])
    f0 = fm_y(mesh.nodes)
    Q = exp_map(np.array([0.0, angle, 0.0]))
    x = (Q @ mesh.nodes.T).T + np.array([0.4, -0.1, 0.7])
    f = fm_y(x)
    assert np.allclose(f, (Q @ f0.T).T, atol=1e-10 * np.abs(f0).max())
    assert np.allclose(_couple(fm_y, x, f), [0.0, 2.0, 0.0])


def test_the_force_is_minus_the_gradient_of_its_potential(face):
    """On a face that is rotated 200 degrees, stretched and randomly warped:
    f_i = -dV/dx_i with V = -|M| theta(x), to 1e-9."""
    mesh, fm, M = face
    rng = np.random.default_rng(0)
    x = (exp_map(np.array([0.2, 3.5, -0.3])) @ (mesh.nodes * [1.0, 1.08, 0.97]).T).T
    x += 0.003 * rng.standard_normal(x.shape)
    f = fm(x)
    h = 1e-6
    for i in fm.idx[::3]:
        for k in range(3):
            xp = x.copy(); xp[i, k] += h
            xm = x.copy(); xm[i, k] -= h
            dV = -np.linalg.norm(M) * (fm.angle(xp) - fm.angle(xm))
            assert abs(f[i, k] + dV / (2 * h)) < 1e-8 * np.abs(f).max()
    assert np.abs(f.sum(axis=0)).max() < 1e-12 * np.abs(f).max()


def test_the_couple_is_smooth_through_a_full_turn_about_a_tilted_axis(face):
    """A face turning 400 degrees about an axis 1 degree off M's. The
    potential -M . phi of the rotation vector jumped at 180 degrees (the
    principal branch flips) and, lifted, was singular at 360 (the tangent
    map's transverse part), which the explicit step turned into a
    sign-alternating tilt oscillation growing 25% a step at 349 degrees.
    The angle about the fixed axis has neither: the force changes by no
    more than its rate times the step everywhere."""
    mesh, _, _ = face
    fm = FaceMoment(mesh.nodes, mesh.elements, "hexa8", mesh.regions["tip"], [0.0, 2.0, 0.0])
    axis = np.array([0.0175, 1.0, 0.0]); axis /= np.linalg.norm(axis)
    prev = None
    for ang in np.arange(0.0, 7.0, 0.01):
        f = fm((exp_map(axis * ang) @ mesh.nodes.T).T)
        if prev is not None:
            assert np.abs(f - prev).max() < 0.015 * np.abs(f).max(), ang
        prev = f
    # the angle itself: the face has turned 7 rad about M's axis to within
    # the tilt, read back unwrapped
    assert abs(((fm.angle((exp_map(axis * 7.0) @ mesh.nodes.T).T) - 7.0) + np.pi) % (2 * np.pi) - np.pi) < 0.02


def test_a_loaded_face_needs_a_plane(face):
    mesh, _, _ = face
    with pytest.raises(ValueError, match="no complete exterior faces"):
        FaceMoment(mesh.nodes, mesh.elements, "hexa8", np.array([0, 1]), [0, 1, 0])


def test_a_short_roll_matches_the_elastica():
    """m = M L / (2 pi E I) = 0.1: the elastica tip u/L = sin(2 pi m)/(2 pi m) - 1,
    w/L = -(1 - cos(2 pi m))/(2 pi m). Sixteen reduced elements along, 2 x 4
    across: 1.5% and 0.7%. (Eight along -- elements ten times longer than
    high -- is 10% soft, the hourglass control's limit, not the load's.)"""
    m = 0.1
    EI = 1.92e6 * 0.05**4 / 12.0
    r = run_case(_bar(divisions=(16, 2, 4), moment=(0.0, 2 * np.pi * m * EI, 0.0)))
    assert r.status == "CONVERGED"
    th = 2 * np.pi * m
    u_ref = np.sin(th) / th - 1.0
    w_ref = -(1.0 - np.cos(th)) / th
    assert abs(r.monitors["u"][-1] / u_ref - 1.0) < 0.03
    assert abs(r.monitors["w"][-1] / w_ref - 1.0) < 0.03


def test_the_dead_load_keeps_its_direction_and_the_follower_its_couple():
    """Without `follower` the reference forces are frozen (a dead load):
    the same first-level answer, and on a turned face a couple that has
    dropped by the cosine of the rotation."""
    pr = prepare_case(_bar())
    mesh = pr.mesh
    fm = FaceMoment(mesh.nodes, mesh.elements, "hexa8", mesh.regions["tip"], [0.0, 1.0, 0.0])
    Q = exp_map(np.array([0.0, 1.0, 0.0]))
    x = (Q @ mesh.nodes.T).T
    dead = fm.reference_forces()
    assert np.isclose(_couple(fm, x, dead)[1], np.cos(1.0))
    assert np.isclose(_couple(fm, x, fm(x))[1], 1.0)


def test_the_eigenvalue_mass_bounds_the_stiffest_mode_in_every_orientation():
    """Hughes' element-eigenvalue inequality: with m = lambda_max(K_e) dt^2 / (4 s^2)
    per element node the assembled system has omega^2 <= 4 s^2 (dt = 1),
    and the mass does not change when the mesh is turned."""
    import scipy.linalg as sla
    from vfife.cli import _with_element, make_solver
    from vfife.elements import continuum

    def solver_for(case):
        case = dict(case); case["mesh"] = dict(case["mesh"]); case["mesh"]["path"] = None
        pr = prepare_case(case)
        return pr, make_solver(_with_element(pr.case, pr.element, pr.case_path), pr.mesh, pr.material)

    case = _bar(divisions=(3, 2, 2), mass="eigenvalue", relaxation_safety=0.5)
    pr, s = solver_for(case)
    b = s.blocks[0]
    lam = continuum.stiffness_max_eigenvalues(pr.mesh.nodes, b.conn, b.pre, b.material.D0)
    assert lam.shape == (12,) and (lam > 0).all()
    # the assembled stiffness at u = 0 by finite differences of the element force
    n = 3 * s.nn
    def force(u):
        fe, _ = b.force(u, b.quat.copy(), b.state.copy(), 1.0, rot_iters=30)
        return b.force.assemble(fe).ravel().copy()
    K = np.empty((n, n)); h = 1e-7
    for j in range(n):
        up = np.zeros(n); up[j] = h
        K[:, j] = (force(up.reshape(-1, 3)) - force(-up.reshape(-1, 3))) / (2 * h)
    K = 0.5 * (K + K.T)
    w2 = sla.eigh(K, np.diag(s.mass.ravel()), eigvals_only=True)
    assert w2[-1] <= 4 * 0.5**2 * (1 + 1e-9)
    # the same mesh, turned: the same eigenvalues to round-off (the row
    # sums of the turned mesh differ by up to a factor of two)
    Q = exp_map(np.array([0.4, -1.1, 0.7]))
    turned = (Q @ pr.mesh.nodes.T).T
    pre2 = continuum.precompute(turned, b.conn, "hexa8", "reduced", material=b.material)
    lam2 = continuum.stiffness_max_eigenvalues(turned, b.conn, pre2, b.material.D0)
    assert np.allclose(lam2, lam, rtol=1e-9)
    rs1 = np.abs(continuum.stiffness_row_sums(pr.mesh.nodes, b.conn, b.pre, b.material.D0))
    rs2 = np.abs(continuum.stiffness_row_sums(turned, b.conn, pre2, b.material.D0))
    assert not np.allclose(rs1, rs2, rtol=1e-3)
