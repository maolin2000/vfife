"""The shell4 path, batched over elements, on the benchmarks that need it.

THE RUN THIS EXISTS FOR
    cases/case_17 (the hinged cylindrical roof of Sze, Liu & Lo 2004) traces
    a full snap-through under displacement control on 256 shell4 elements;
    their pinched hemisphere (parked in cases/scratch) pinches
    256 of them into an ellipse. The element-at-a-time loop the shell solver
    had -- the same shape the frame solver had until v0.1 -- ran
    those at a few iterations a second. The batched version is 44x faster
    and is pinned here to the loop it replaced; the numba
    kernel and the warp backend are pinned to the batched version.
"""
import pathlib
import tempfile

import numpy as np
import pytest

from vfife.api import prepare_case, run_case
from vfife.core.rotations import exp_map, log_map
from vfife.cli import _with_element, make_solver
from benchmarks.generate_sze_shells import roof


def _roof_problem(n=4, target=-12.0, levels=12):
    d = pathlib.Path(tempfile.mkdtemp())
    roof(d / "roof.msh", n)
    return {
        "name": "roof_test", "units": {"length": "mm", "force": "N", "stress": "MPa"},
        "mesh": {"type": "gmsh", "path": str(d / "roof.msh")},
        "materials": {"roof": {"type": "linear_elastic", "E": 3102.75, "nu": 0.3, "rho": 3.1e-9}},
        "sections": {"shell": {"type": "shell", "material": "roof", "thickness": 12.7,
                               "shear_correction": 0.8333333333, "drilling_factor": 1.0,
                               "hourglass_coeff": 0.1, "formulation": "selective"}},
        "assign": [{"region": "all", "section": "shell"}],
        "constraints": [{"region": "sym_x", "dofs": [1, 0, 0, 0, 1, 1]},
                        {"region": "sym_y", "dofs": [0, 1, 0, 1, 0, 1]},
                        {"region": "hinged", "dofs": [1, 1, 1, 0, 0, 0]}],
        "steps": [{"name": "push", "constraints": [
            {"region": "crown", "dofs": [0, 0, 1, 0, 0, 0], "value": target}],
            "load_steps": levels, "max_steps_per_load": 20000, "tolerance": 1.0e-5}],
        "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 240000,
                   "damping": {"type": "kinetic"}, "tolerance": 1.0e-5},
        "output": {"path": str(d / "r.h5"), "every": 10**7,
                   "monitors": {"w_C": {"kind": "displacement", "region": "crown", "dof": "z"},
                                "P4": {"kind": "reaction", "region": "crown", "dof": "z"}}},
    }


def _solver(problem):
    pr = prepare_case(problem)
    return make_solver(_with_element(pr.case, pr.element, pr.case_path), pr.mesh, pr.material)


def _smooth_field(s, amp):
    """A rigid rotation of `amp` radians about x plus a bowl of the same
    order: large rotations everywhere, no element warped."""
    X = s.mesh.nodes
    Rx = np.array([[1, 0, 0], [0, np.cos(amp), -np.sin(amp)], [0, np.sin(amp), np.cos(amp)]])
    x = (Rx @ X.T).T
    x[:, 2] += amp * 0.05 * X[:, 0] * X[:, 1] / (X[:, 0].max() * X[:, 1].max() + 1e-30)
    q = np.zeros((s.nn, 6)); q[:, :3] = x - X; q[:, 3] = amp
    s.orient = exp_map(q[:, 3:])                  # the state is the orientation
    return q.ravel()


@pytest.mark.parametrize("amp", [0.01, 1.5, 3.0])
def test_batched_shell4_forces_equal_the_reference_loop(amp):
    s = _solver(_roof_problem(6))
    s.q = _smooth_field(s, amp)
    f_b, r_b = s.internal_force()
    f_r, r_r = s._internal_force_reference()
    # 1e-9 of the assembled force: the field is nearly rigid, so the nodal
    # forces are a small remainder of large element contributions that
    # cancel between neighbours, and the two summation orders differ at the
    # level of that cancellation
    assert np.max(np.abs(f_b - f_r)) < 1e-9 * np.max(np.abs(f_r))
    assert np.max(np.abs(r_b - r_r)) < 1e-9 * np.max(np.abs(r_r))


def test_the_refreshed_shell_mass_is_the_setup_mass_at_the_reference():
    s = _solver(_roof_problem(6))
    m0 = s.mass.copy()
    s._refresh_fictitious_mass()
    assert np.allclose(s.mass, m0, rtol=1e-12, atol=0.0)


def test_a_warped_current_face_is_still_refused():
    """The loop raised on a warped face; so must the batch, or a folded
    element would quietly produce numbers."""
    s = _solver(_roof_problem(4))
    conn = s.mesh.elements[0]
    s.q[:] = 0.0
    s.q[s.dofs.node_dofs(conn[0])[0, 2]] = 200.0          # one corner lifted 200 mm
    with pytest.raises(ValueError, match="warped"):
        s.internal_force()


def test_the_roof_snaps_through_its_limit_point_under_displacement_control():
    """4 x 4 on the quarter, driven 12 mm: enough to pass the limit point
    Sze puts at 0.7421 P_max, w_C = 11.3. The coarse mesh gives 0.752 at
    11.0 and is already descending at 12."""
    r = run_case(_roof_problem(4, -12.0, 12))
    w = -np.asarray(r.monitors["w_C"]); P = -4.0 * np.asarray(r.monitors["P4"]) / 3000.0
    i = int(np.argmax(P))
    assert abs(P[i] / 0.7421 - 1.0) < 0.03, P
    assert 10.0 <= w[i] <= 12.0
    assert P[-1] < P[i], "the path must be past the limit point"


@pytest.mark.parametrize("amp", [0.01, 1.5, 3.0])
def test_the_numba_shell4_kernel_equals_the_numpy_path(amp):
    pytest.importorskip("numba")
    s = _solver(_roof_problem(6))
    s.q = _smooth_field(s, amp)
    f_ref, r_ref = s._internal_force_numpy()
    f_nb, r_nb = s._internal_force_numba()
    assert np.max(np.abs(f_nb - f_ref)) < 1e-9 * np.max(np.abs(f_ref))
    assert np.max(np.abs(r_nb - r_ref)) < 1e-9 * np.max(np.abs(r_ref))


def test_the_numba_kernel_refuses_a_warped_face():
    pytest.importorskip("numba")
    s = _solver(_roof_problem(4))
    conn = s.mesh.elements[0]
    s.q[:] = 0.0; s.q[s.dofs.node_dofs(conn[0])[0, 2]] = 200.0
    with pytest.raises(ValueError, match="warped"):
        s._internal_force_numba()


# ---------------------------------------------------------------------- #
# the warp backend: the same arithmetic on a device      #
# ---------------------------------------------------------------------- #

warp = pytest.importorskip("warp", reason="the warp backend needs warp-lang")


def _warp_solver(problem, precision="fp64"):
    problem["solver"].update(backend="warp", device="cpu", precision=precision)
    s = _solver(problem)
    return s, s._make_warp_shell()


@pytest.mark.parametrize("precision, tol", [("fp64", 1e-9), ("mixed", 1e-6)])
def test_the_warp_shell4_kernel_equals_the_numpy_path(precision, tol):
    """fp64 to the summation order; mixed keeps the 24x24 local stiffness
    in fp32, and 1e-7 in its entries is 1e-7 in the force."""
    s, ws = _warp_solver(_roof_problem(6), precision)
    s.q = _smooth_field(s, 1.5)
    f_ref, r_ref = s._internal_force_numpy()
    ws.set_u(s.q, None, s.orient); ws.force_resident(); ws.check_flags()
    assert np.max(np.abs(ws.get_F_int() - f_ref)) < tol * np.max(np.abs(f_ref))
    assert np.max(np.abs(ws.get_result() - r_ref)) < tol * np.max(np.abs(r_ref))


def test_the_warp_kernel_refuses_a_warped_face_in_the_same_readback():
    """A refused face is counted into the per-iteration scalars, so the
    resident loop learns of it on the iteration it happens rather than at
    the end of the level, and the message is the host's."""
    s, ws = _warp_solver(_roof_problem(4))
    conn = s.mesh.elements[0]
    s.q[:] = 0.0; s.q[s.dofs.node_dofs(conn[0])[0, 2]] = 200.0
    ws.set_u(s.q); ws.force_resident()
    fixed, _m, prescribed, groups, _g = s.step_data(0)
    ws.set_nodal(fixed, groups, [(d, v, a) for d, v, a, _ in prescribed])
    assert ws.measures(1.0)[5] == 1
    with pytest.raises(ValueError, match="warped"):
        ws.check_flags()


def test_the_resident_loop_stops_where_the_host_loop_stops():
    """Same staged relaxation, same kinetic-damping resets, same
    every-iteration check: the device run must converge at the same
    iteration as the numpy run and give the same curve. fp64 pins that to
    round-off; mixed to the fp32 stiffness."""
    host = run_case(_roof_problem(4, -4.0, 4))
    for precision, tol in (("fp64", 1e-10), ("mixed", 1e-6)):
        problem = _roof_problem(4, -4.0, 4)
        problem["solver"].update(backend="warp", device="cpu", precision=precision)
        dev = run_case(problem)
        assert dev.converged
        assert dev.diagnostics["step"] == host.diagnostics["step"], precision
        P_h = np.asarray(host.monitors["P4"]); P_d = np.asarray(dev.monitors["P4"])
        assert np.max(np.abs(P_d - P_h)) < tol * np.max(np.abs(P_h)), precision


def test_a_machine_without_cuda_gets_a_sentence_not_a_stack_trace():
    from vfife.core.diagnostics import UserError
    if warp.is_cuda_available():
        pytest.skip("this machine has CUDA")
    problem = _roof_problem(4, -1.0, 2)
    problem["solver"].update(backend="warp", device="cuda")
    with pytest.raises(UserError, match="BACKEND_UNAVAILABLE|cannot provide"):
        run_case(problem)


def test_a_rigidly_moved_roof_carries_no_force():
    """Every backend: the whole roof rotated by a large skew rotation and
    translated, every point's orientation the same rotation -- no force.
    Until v0.1 the shell subtracted rotation vectors and a
    non-coaxial rigid rotation left a residue."""
    Q = exp_map(np.array([0.7, -0.4, 1.1]))
    for backend in ("numpy", "numba"):
        p = _roof_problem(6); p["solver"]["backend"] = backend
        s = _solver(p)
        X = s.mesh.nodes; x = (Q @ X.T).T + 5.0
        q = np.zeros((s.nn, 6)); q[:, :3] = x - X; q[:, 3:] = log_map(Q)
        s.q = q.ravel(); s.orient = np.broadcast_to(Q, (s.nn, 3, 3)).copy()
        f, _ = s.internal_force()
        assert np.abs(f).max() < 1e-12 * np.abs(s._sh_K).max(), backend
