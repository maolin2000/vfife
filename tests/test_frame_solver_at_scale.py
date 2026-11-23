"""What Lee's frame found in the frame solver, pinned so it stays found.

THE RUN THIS EXISTS FOR
    cases/case_16 is forty co-rotational beam elements driven through a limit
    point, a snap-back and a load reversal. Every beam case before it had one
    or two elements and a load. Its first run exposed four things, none of
    them mechanics:

      * the member loop ran in Python at 120 iterations/s;
      * the residual was normalised by max(|F_ext|, 1 N), an absolute
        newton count under displacement control;
      * a step's own `load_steps` was ignored, and a prescribed displacement
        jumped to each level's target in one iteration;
      * the fictitious mass came from the reference stiffness and stopped
        being a stability bound once the members had rotated.

    The mechanics, once those were out of the way, hit NAFEMS NL7's limit
    load to 0.09% (verification/case_16_lee_frame.yaml).
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case, run_case
from vfife.cli import _with_element, make_solver

E, L = 71.74e9, 1.2


def _frame_problem(n=10):
    """Lee's frame as an inline case, `n` elements per member."""
    from benchmarks.generate_lee_frame import build
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    build(d / "lee.msh", n)
    return {
        "name": "lee_test", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": str(d / "lee.msh")},
        "materials": {"al": {"type": "linear_elastic", "E": E, "nu": 0.0, "rho": 2700.0}},
        "sections": {"b": {"type": "beam", "material": "al", "shape": "general",
                           "area": 6.0e-4, "iy": 2.0e-8, "iz": 2.0e-8, "j": 4.0e-8}},
        "assign": [{"region": "members", "section": "b", "orientation": [0, 1, 0]}],
        "constraints": [{"region": "all", "dofs": [0, 1, 0, 1, 0, 1]},
                        {"region": "pin_l", "dofs": [1, 1, 1, 1, 0, 1]},
                        {"region": "pin_r", "dofs": [1, 1, 1, 1, 0, 1]}],
        "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 200000,
                   "damping": {"type": "kinetic"}, "tolerance": 1.0e-5},
        "output": {"path": str(d / "r.h5"), "every": 10**7,
                   "monitors": {"v_A": {"kind": "displacement", "region": "load", "dof": "z"},
                                "P": {"kind": "reaction", "region": "load", "dof": "z"}}},
    }


def _solver(problem):
    pr = prepare_case(problem)
    return make_solver(_with_element(pr.case, pr.element, pr.case_path), pr.mesh, pr.material)


# --------------------------------------------------------------------------- #
#  1. the batched member force is the element-at-a-time one
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("amplitude", [1e-3, 0.3])
def test_batched_member_forces_equal_the_reference_loop(amplitude):
    """Random translations AND rotations of every node, up to 0.3 m and 0.3
    rad, so every member has a different rigid rotation to remove."""
    s = _solver(_frame_problem(10))
    rng = np.random.default_rng(0)
    s.q = rng.normal(size=s.q.shape) * amplitude
    s.sync_orientation()
    f_batched, a_batched = s.internal_force()
    f_ref, a_ref = s._internal_force_reference()
    scale = np.max(np.abs(f_ref))
    assert np.max(np.abs(f_batched - f_ref)) < 1e-12 * scale
    assert np.max(np.abs(a_batched - a_ref)) < 1e-12 * np.max(np.abs(a_ref))


def test_a_member_flipped_through_180_degrees_takes_the_fallback_axis():
    """The scalar version has a branch for s < 1e-12 with c < 0: the chord has
    reversed and the rotation axis is undefined, so it is taken from the
    reference frame's second direction. The batched one must agree there too,
    and stay finite."""
    s = _solver(_frame_problem(10))
    conn = s.mesh.elements[3]; dof = s.dofs.element_dofs(conn)
    dz = s.mesh.nodes[conn[1], 2] - s.mesh.nodes[conn[0], 2]
    s.q[:] = 0.0; s.q[dof[8]] = -2.0 * dz          # far node moved to the mirror position
    f_b, _ = s.internal_force(); f_r, _ = s._internal_force_reference()
    assert np.all(np.isfinite(f_b))
    assert np.max(np.abs(f_b - f_r)) < 1e-12 * np.max(np.abs(f_r))


# --------------------------------------------------------------------------- #
#  2. the residual under displacement control is relative to something
# --------------------------------------------------------------------------- #

def test_the_residual_scale_is_the_internal_force_when_there_is_no_load():
    """Under pure displacement control F_ext is zero on every free dof. The
    scale must then be the internal force (the reactions), not 1 N."""
    s = _solver(_frame_problem(10))
    rng = np.random.default_rng(1)
    s.q = rng.normal(size=s.q.shape) * 1e-3
    s.sync_orientation()
    F = np.zeros(s.ndof); free = np.ones(s.ndof, dtype=bool)
    f_int = s.internal_force()[0]
    resid, _kr = s._convergence_measures(F, free, 0.0)
    assert np.isclose(resid, 1.0, rtol=1e-12), (
        "with F_ext = 0 the free-dof residual IS the internal force, so the "
        "ratio must come out as exactly one -- it used to be |F_int| / 1 N")
    assert np.linalg.norm(f_int) > 1e3, "the state is not trivial"


# --------------------------------------------------------------------------- #
#  3. a step's own staging is honoured, and the displacement ramps in
# --------------------------------------------------------------------------- #

def test_a_steps_own_load_steps_are_honoured():
    """Fifty levels asked for in the STEP, none in the solver block: fifty
    converged frames must come back, not one."""
    p = _frame_problem(5)
    p["steps"] = [{"name": "push", "constraints": [
        {"region": "load", "dofs": [0, 0, 1, 0, 0, 0], "value": -0.05}],
        "load_steps": 5, "max_steps_per_load": 30000, "tolerance": 1.0e-4}]
    r = run_case(p)
    assert len(r.monitors["P"]) == 5
    v = np.asarray(r.monitors["v_A"])
    assert np.allclose(v, -0.01 * np.arange(1, 6), atol=1e-12)


def test_a_prescribed_displacement_approaches_its_level_target():
    """The first iterations of a level move the driven dof part of the way,
    not all of it: the frame solver ramps like the solid solver does."""
    s = _solver(_frame_problem(5))
    s.steps = [{"name": "push", "constraints": [
        {"region": "load", "dofs": [0, 0, 1, 0, 0, 0], "value": -0.1,
         "motion": "displacement", "amplitude": None}], "loads": []}]
    fixed, moving, prescribed, groups, ground = s.step_data(0)
    dofs = prescribed[0][0]
    start = [np.zeros(len(dofs))]
    s._apply_bc(fixed, moving, prescribed, 1.0, start, approach=0.25)
    assert np.allclose(s.q[dofs], -0.025)
    s._apply_bc(fixed, moving, prescribed, 1.0, start, approach=1.0)
    assert np.allclose(s.q[dofs], -0.1)


# --------------------------------------------------------------------------- #
#  4. the fictitious mass follows the geometry
# --------------------------------------------------------------------------- #

def test_the_refreshed_mass_is_the_setup_mass_at_the_reference():
    s = _solver(_frame_problem(10))
    m0 = s.mass.copy()
    s._refresh_fictitious_mass()
    assert np.array_equal(s.mass, m0)


def test_the_refreshed_mass_grows_where_members_have_rotated():
    """Rotate the beam member 90 degrees about the corner (rigidly): its
    axial stiffness now sits in the z rows of its nodes, whose reference
    row sums held only the bending terms. The ratio is (EA/L) / (12EI/L^3)
    = A L^2 / 12 I = 36 on this mesh; a mass that did not follow would leave
    those dofs 36-fold too light and the time step six times over the
    stability limit."""
    s = _solver(_frame_problem(10))
    m0 = s.mass.copy()
    n = 10
    for i in range(1, n + 1):                    # beam nodes n+1 .. 2n
        node = n + i
        x0 = s.mesh.nodes[node]
        # (x, z) = (L i/n, L) -> (0, L + L i/n): a rigid 90-degree swing up
        s.q[s.dofs.node_dofs(node)[0, 0]] = -x0[0]
        s.q[s.dofs.node_dofs(node)[0, 2]] = x0[0]
        s.q[s.dofs.node_dofs(node)[0, 4]] = np.pi / 2
    s.sync_orientation()
    s._refresh_fictitious_mass()
    z_dofs = [s.dofs.node_dofs(n + i)[0, 2] for i in range(1, n)]
    assert np.all(s.mass[z_dofs] > 20.0 * m0[z_dofs])


# --------------------------------------------------------------------------- #
#  5. and the mechanics: the limit load, on a coarse mesh, quickly
# --------------------------------------------------------------------------- #

def test_the_limit_load_on_a_coarse_mesh_is_within_two_percent_of_nafems():
    """Ten elements a member, driven to v/L = 0.45 -- past the limit point,
    short of the snap-back. NAFEMS: PL^2/EI = 18.552 at v/L = 0.407."""
    p = _frame_problem(10)
    p["steps"] = [{"name": "push", "constraints": [
        {"region": "load", "dofs": [0, 0, 1, 0, 0, 0], "value": -0.54}],
        "load_steps": 9, "max_steps_per_load": 40000, "tolerance": 1.0e-5}]
    r = run_case(p)
    P = -np.asarray(r.monitors["P"]); v = -np.asarray(r.monitors["v_A"])
    lam = P * L * L / (E * 2.0e-8)
    i = int(np.argmax(lam))
    assert abs(lam[i] / 18.552 - 1.0) < 0.02, lam
    assert 0.35 < v[i] / L < 0.47
    assert lam[-1] < lam[i], "the path must be past the limit point"


def test_a_shell_line_relaxation_mass_follows_both_halves_when_turned():
    """The shell's refresh was for `shell4` only, so a shell_line model kept
    its reference rows through every load level. At the
    reference the refresh is the setup mass; turned rigidly, it is the
    rows of the turned setup matrix, line rows included -- the beam-only
    node's lateral row takes up the axial stiffness."""
    from scipy import sparse
    from test_shell4 import problem, NODES, E, NU
    from vfife.core.shell_line_solver import ShellLineSolver
    from vfife.core.mesh import Mesh
    from vfife.materials import create
    from scipy.spatial.transform import Rotation
    blocks = {"shell4": np.array([[0, 1, 2, 3]], np.int32),
              "truss2": np.array([[0, 1], [1, 4]], np.int32)}
    nodes = np.vstack([NODES, [[3., 0, 0]]])
    mesh = Mesh(nodes, blocks["shell4"], {"all": np.arange(5), "beam": np.array([0, 1, 4])}, blocks)
    p = problem(); p["element"]["type"] = "shell_line"
    p["members"] = [{"region": "beam", "type": "beam2", "area": .01, "iy": 1e-5, "iz": 2e-5,
                     "j": 3e-5, "orientation": [0, 0, 1], "releases": []}]
    s = ShellLineSolver(p, mesh, create("linear_elastic", E=E, nu=NU, rho=7850))
    setup = s.mass.copy(); K0 = s.K.copy()
    s._refresh_fictitious_mass()
    assert s.mass == pytest.approx(setup, rel=1e-12)
    R = Rotation.from_rotvec([0.1, -0.2, 0.5]).as_matrix()
    q = s.q.reshape(-1, 6); X = s.mesh.nodes
    q[:, :3] = X @ R.T - X; s.orient[:] = R
    s._refresh_fictitious_mass()
    T = sparse.block_diag([R] * (2 * s.nn), format="csr")
    want = np.asarray(abs(T @ K0 @ T.T).sum(axis=1)).ravel() / float(s.p.solver.relaxation_safety)**2
    assert s.mass == pytest.approx(want, rel=1e-12)
    assert s.mass[6*4+1] > 10 * setup[6*4+1]
