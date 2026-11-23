"""A floor that is rigid in its own plane.

The one multi-point constraint a building cannot be written without: the slab
is not meshed, so without it the floor beams carry the in-plane stiffness of a
frame where a plate belongs, and there is no single number per floor to compare
against an OpenSees `rigidDiaphragm` model's master node.

What is asserted here is not "the numbers match a reference" -- there is no
reference -- but the four things a constraint of this kind is: the floor moves
rigidly in its plane, it does NOT move rigidly out of it, the condensation
conserves the resultants it condenses, and a floor with nothing to constrain
(one already rigid) is left where it was.
"""

import numpy as np
import pytest

from vfife.testing import models


def _frame(diaphragms=None, **kw):
    """A two-storey frame: four columns, a beam ring at each level."""
    z = [0.0, 3.0, 6.0]
    plan = [(0.0, 0.0), (4.0, 0.0), (4.0, 5.0), (0.0, 5.0)]
    nodes, index = [], {}
    for k, zz in enumerate(z):
        for j, (x, y) in enumerate(plan):
            index[(k, j)] = len(nodes); nodes.append([x, y, zz])
    nodes = np.array(nodes, dtype=float)
    lines = []
    for k in range(len(z) - 1):
        for j in range(4):
            lines.append([index[(k, j)], index[(k + 1, j)]])
    for k in (1, 2):
        for j in range(4):
            lines.append([index[(k, j)], index[(k, (j + 1) % 4)]])
    return nodes, np.array(lines, dtype=np.int32), index


def _solver(diaphragm=True, mass=1.0e4):
    from vfife.cli import make_solver
    from vfife.core.mesh import Mesh
    from vfife.materials.factory import create_material

    nodes, lines, index = _frame()
    mesh = Mesh(nodes=nodes, elements=lines, cell_blocks={"truss2": lines})
    base = np.array([index[(0, j)] for j in range(4)])
    floors = [np.array([index[(k, j)] for j in range(4)]) for k in (1, 2)]
    mesh.regions = {"base": base, "floor1": floors[0], "floor2": floors[1],
                    "all": np.arange(len(nodes)), "top": floors[1],
                    "columns": np.arange(len(nodes)), "beams": np.arange(len(nodes))}
    # element regions, for the same reason a building needs them: every
    # column's two nodes are beam nodes too (core/mesh.py::element_regions)
    mesh.element_regions = {"columns": {"truss2": np.arange(8)},
                            "beams": {"truss2": np.arange(8, len(lines))}}
    section = {"type": "beam2", "area": 0.16, "iy": 2.1e-3, "iz": 2.1e-3, "j": 3.6e-3}
    problem = {
        "element": {"type": "mixed_line", "formulation": "full"},
        "material": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2400.0},
        "members": [{"region": "columns", "orientation": [1.0, 0.0, 0.0], **section},
                    {"region": "beams", "orientation": [0.0, 0.0, 1.0], **section}],
        "constraints": [{"region": "base", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": [],
        "nodal_masses": [{"region": "floor1", "mass": mass, "per_node": True},
                         {"region": "floor2", "mass": mass, "per_node": True}],
        "solver": {"mode": "dynamic", "backend": "numpy", "total_time": 0.2,
                   "dt_safety": 0.5, "damping": {"type": "viscous", "coeff": 0.0}},
        "output": {"monitors": {}},
        "steps": [{}],
    }
    if diaphragm:
        problem["diaphragms"] = [{"region": "floor1", "normal": "z"},
                                 {"region": "floor2", "normal": "z"}]
    return (make_solver(problem, mesh, create_material(problem["material"])),
            mesh, floors)


def _accel(s, force):
    """The acceleration field the solver would use for `force`."""
    return s._project_diaphragms(force / s.mass)


def _rigid_residual(x, acc, idx):
    """How far the floor's in-plane acceleration is from a rigid motion."""
    p = x[idx, :2] - x[idx, :2].mean(axis=0)
    u = acc.reshape(-1, 6)[idx][:, :2]
    A = np.zeros((2 * len(idx), 3)); b = np.zeros(2 * len(idx))
    A[0::2, 0] = 1; A[0::2, 2] = -p[:, 1]; b[0::2] = u[:, 0]
    A[1::2, 1] = 1; A[1::2, 2] = p[:, 0]; b[1::2] = u[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = b - A @ sol
    return float(np.linalg.norm(res) / max(np.linalg.norm(b), 1e-300))


def test_the_floor_moves_rigidly_in_its_own_plane():
    s, mesh, floors = _solver()
    rng = np.random.default_rng(0)
    force = rng.normal(scale=1.0e5, size=s.ndof)
    acc = _accel(s, force)
    for k, idx in enumerate(floors):
        assert _rigid_residual(mesh.nodes, acc, idx) < 1e-12, k


def test_without_the_diaphragm_it_does_not():
    """The control. A check that cannot fail is not a check."""
    s, mesh, floors = _solver(diaphragm=False)
    rng = np.random.default_rng(0)
    acc = _accel(s, rng.normal(scale=1.0e5, size=s.ndof))
    assert _rigid_residual(mesh.nodes, acc, floors[0]) > 0.1


def test_the_floor_is_free_out_of_its_plane():
    """A diaphragm is not a rigid body: the vertical translation and the two
    bending rotations pass through untouched."""
    s, mesh, floors = _solver()
    rng = np.random.default_rng(1)
    force = rng.normal(scale=1.0e5, size=s.ndof)
    before = (force / s.mass).reshape(-1, 6)
    after = _accel(s, force).reshape(-1, 6)
    for idx in floors:
        for c in (2, 3, 4):                      # uz, rx, ry
            assert np.allclose(after[idx, c], before[idx, c]), c


def test_the_condensation_conserves_the_resultants():
    """The rigid floor is accelerated by exactly the force and torque its
    nodes carried: the constraint moves the motion about, it does not add or
    remove any."""
    s, mesh, floors = _solver()
    rng = np.random.default_rng(2)
    force = rng.normal(scale=1.0e5, size=s.ndof)
    before = (force / s.mass)
    after = _accel(s, force.copy())
    m = s.mass.reshape(-1, 6)
    for idx in floors:
        x = mesh.nodes[idx, :2] - mesh.nodes[idx, :2].mean(axis=0)
        for field in (before, after):
            f = m[idx, :2] * field.reshape(-1, 6)[idx, :2]
            mz = (m[idx, 5] * field.reshape(-1, 6)[idx, 5]).sum() \
                + (x[:, 0] * f[:, 1] - x[:, 1] * f[:, 0]).sum()
            if field is before:
                want = (f.sum(axis=0), mz)
            else:
                assert np.allclose(f.sum(axis=0), want[0], rtol=1e-10)
                assert np.isclose(mz, want[1], rtol=1e-10)


def test_a_motion_that_is_already_rigid_is_left_alone():
    """Give the floor exactly the acceleration field a rigid motion would
    have, and the projection returns it unchanged -- so the constraint is
    idempotent, which a projection has to be."""
    s, mesh, floors = _solver()
    field = np.zeros((s.nn, 6))
    for idx in floors:
        x = mesh.nodes[idx, :2] - mesh.nodes[idx, :2].mean(axis=0)
        Ax, Ay, alpha = 0.7, -0.3, 0.05
        field[idx, 0] = Ax - alpha * x[:, 1]
        field[idx, 1] = Ay + alpha * x[:, 0]
        field[idx, 5] = alpha
    force = (field.reshape(-1) * s.mass)
    after = _accel(s, force).reshape(-1, 6)
    for idx in floors:
        assert np.allclose(after[idx][:, (0, 1, 5)], field[idx][:, (0, 1, 5)], atol=1e-12)


def test_a_floor_that_is_also_held_by_a_support_is_refused():
    """Either a support holds the floor or a diaphragm moves it. Both at once
    is a constraint the projection cannot satisfy, and it says so at setup
    rather than quietly losing to whichever runs last."""
    s, mesh, floors = _solver()
    # the solver's case is read-only (core/problem.py): a changed copy
    s.p = s.p.derive(diaphragms=[{"region": "base", "normal": "z"}])
    s._diaphragms = None
    with pytest.raises(ValueError, match="constrained"):
        s._project_diaphragms(np.zeros(s.ndof))


def test_a_load_on_one_corner_moves_the_whole_storey():
    """The engineering statement, end to end. A force applied at one corner
    of the top floor accelerates only that corner when the slab is not there;
    with the diaphragm the storey takes it together.

    NOT "the floor moves as one block": a rigid floor loaded off its centre
    also SPINS, so the nodal accelerations still differ across it. What
    changes is that none of them is left behind.
    """
    out = {}
    for tag in (True, False):
        s, mesh, floors = _solver(diaphragm=tag)
        top = floors[1]
        F = np.zeros(s.ndof); F[6 * top[0]] = 4.0e5
        acc = _accel(s, F).reshape(-1, 6)[top]
        speed = np.linalg.norm(acc[:, :2], axis=1)
        out[tag] = speed.min() / speed.max()
    assert out[False] < 1e-12, out
    assert out[True] > 0.2, out
