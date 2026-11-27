"""Linear 3-D frame member regressions."""
from pathlib import Path

import pytest
import numpy as np

from vfife.cli import _with_element, build, make_solver
from vfife.io import schema

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "cases" / "case_01_beam_cantilever.yaml"


def test_beam_line2_import_and_cantilever_matches_euler_bernoulli():
    case, _ = schema.load(str(CASE))
    mesh, material, element = build(case, str(CASE))
    assert element == "beam2"
    solver = make_solver(_with_element(case, element, str(CASE)), mesh, material)
    solver.run(output_every=1, progress_every=1)
    assert solver.u[1, 2] == pytest.approx(-1000.0 / (3.0 * 2e11 * 1e-4), rel=1e-12)
    assert solver.monitors["root_fz"].values[-1] == pytest.approx(1000.0, rel=1e-8)
    fields = solver.nodal_fields(solver._cell_actions())
    # signed since v1.0: the root moment of a tip load P = -1000 in z
    # is (1, 0, 0) x (0, 0, -1000) = +1000 about global y, and local z is
    # x cross [0, 0, 1] = -y, so -1000 about local z; the envelope keeps 1000
    assert fields["bending_z"][0] == pytest.approx(-1000.0, rel=1e-8)
    assert fields["bending_z_max_abs"][0] == pytest.approx(1000.0, rel=1e-8)


def test_beam_schema_requires_section_and_six_dof_constraints():
    with pytest.raises(ValueError, match="iy"):
        schema.ElementSpec(type="beam2", formulation="full", area=1e-3)
    case, _ = schema.load(str(CASE))
    p = _with_element(case, "beam2", str(CASE))
    p["constraints"][0]["dofs"] = [1, 1, 1]
    mesh, material, _ = build(case, str(CASE))
    with pytest.raises(ValueError, match="six DOF"):
        make_solver(p, mesh, material).run()


def test_beam_dynamic_velocity_constraint_uses_six_dof_state():
    case, _ = schema.load(str(CASE))
    p = _with_element(case, "beam2", str(CASE))
    p["solver"] = {**p["solver"], "mode": "dynamic", "total_time": 0.001,
                   "damping": {"type": "none"}}
    p["loads"] = []
    p["constraints"] = [
        {"region": "fixed", "dofs": [1, 1, 1, 1, 1, 1]},
        {"region": "loaded", "dofs": [1, 1, 1, 1, 1, 1],
         "motion": "velocity", "value": [0, 0, 0.2, 0, 0, 0], "amplitude": "hold"},
    ]
    mesh, material, _ = build(case, str(CASE))
    s = make_solver(p, mesh, material)
    s.run(output_every=999999, progress_every=999999)
    assert s.u[1, 2] == pytest.approx(0.2 * s.n_steps * s.dt, rel=1e-12)


def test_corotational_beam_has_no_force_under_rigid_motion():
    case, _ = schema.load(str(CASE)); mesh, material, element = build(case, str(CASE))
    s = make_solver(_with_element(case, element, str(CASE)), mesh, material)
    angle = np.pi / 3; Q = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                      [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    q = s.q.reshape(-1, 6)
    q[:, :3] = mesh.nodes @ Q.T - mesh.nodes + [0.2, -0.1, 0.4]
    q[:, 3:] = [0., 0., angle]
    s.sync_orientation()
    force, actions = s.internal_force()
    # round-off of a 60-degree rotation: 9e-8 N with NumPy 2, 2e-7 N with
    # NumPy 1.24's LAPACK -- against EA = 2e9 N, 1e-16 either way
    assert np.abs(force).max() < 1e-6
    assert np.abs(actions).max() < 1e-6


def test_early_convergence_persists_terminal_diagnostics():
    case, _ = schema.load(str(CASE)); mesh, material, element = build(case, str(CASE))
    s = make_solver(_with_element(case, element, str(CASE)), mesh, material)
    s.run(output_every=999999, progress_every=999999)
    assert len(s.residuals) == len(s.kinetic) == 1
    assert s.residuals[-1] <= s.tolerance
    assert np.isfinite(s.kinetic[-1])


def test_uniform_local_line_load_is_consistently_integrated():
    case, _ = schema.load(str(CASE)); mesh, material, element = build(case, str(CASE))
    p = _with_element(case, element, str(CASE))
    p["loads"] = [{"region": "bar", "line_load": [0, 10, 0],
                   "coordinate_system": "local"}]
    s = make_solver(p, mesh, material)
    _fixed, _moving, _prescribed, groups, _ground = s.step_data(0)
    force = groups[0][1]
    # For this x-axis member orientation, local y is global z. The consistent
    # vector carries qL total force and the equal/opposite qL²/12 end moments.
    assert force.reshape(-1, 6)[:, 2].sum() == pytest.approx(10.0)
    assert force.reshape(-1, 6)[0, 4] == pytest.approx(-10.0/12.0)
    assert force.reshape(-1, 6)[1, 4] == pytest.approx(10.0/12.0)


def test_released_end_gets_no_equivalent_line_load_moment():
    case, _ = schema.load(str(CASE)); mesh, material, element = build(case, str(CASE))
    p = _with_element(case, element, str(CASE))
    p["element"]["releases"] = ["j_rz"]
    p["loads"] = [{"region": "bar", "line_load": [0, 10, 0],
                   "coordinate_system": "local"}]
    s = make_solver(p, mesh, material)
    force = s.step_data(0)[3][0][1].reshape(-1, 6)
    # local rz maps to negative global ry for this orientation.
    assert force[1, 4] == pytest.approx(0.0, abs=1e-14)


def test_signed_i_j_local_actions_are_available():
    case, _ = schema.load(str(CASE)); mesh, material, element = build(case, str(CASE))
    s = make_solver(_with_element(case, element, str(CASE)), mesh, material)
    s.run(output_every=999999, progress_every=999999)
    fields = s.nodal_fields(s._cell_actions())
    # With orientation=[0,0,1], global z is the member's local y direction.
    assert fields["shear_y_i"][0] == pytest.approx(1000.0)
    assert fields["shear_y_j"][0] == pytest.approx(-1000.0)
    assert fields["bending_z_i"][0] == pytest.approx(1000.0)
    assert fields["bending_z_j"][0] == pytest.approx(0.0, abs=1e-6)
