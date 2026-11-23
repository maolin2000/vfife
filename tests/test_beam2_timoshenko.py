"""Qualification tests for the 3-D co-rotational Timoshenko beam."""
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

from vfife.cli import _with_element, build, make_solver
from vfife.io import schema


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cases" / "case_01_beam_cantilever.yaml"


def _solver(**section):
    case, _ = schema.load(str(BASE))
    mesh, material, _ = build(case, str(BASE))
    p = _with_element(case, "beam2_timoshenko", str(BASE))
    p["element"].update({
        "type": "beam2_timoshenko", "ay": 8.0e-3, "az": 6.0e-3,
        "kappa_y": 5.0/6.0, "kappa_z": 0.75,
    })
    p["element"].update(section)
    return make_solver(p, mesh, material)


def _cantilever_linear_solution(solver, force):
    fixed = np.arange(6)
    free = np.setdiff1d(np.arange(solver.ndof), fixed)
    q = np.zeros(solver.ndof)
    q[free] = np.linalg.solve(solver.K[np.ix_(free, free)].toarray(), force[free])
    return q


def test_schema_requires_both_shear_areas_and_keeps_eb_contract_clean():
    common = dict(type="beam2_timoshenko", formulation="full", area=1e-2,
                  iy=1e-4, iz=2e-4, j=3e-4)
    with pytest.raises(ValueError, match="ay and element.az"):
        schema.ElementSpec(**common)
    spec = schema.ElementSpec(**common, ay=8e-3, az=7e-3,
                              kappa_y=5/6, kappa_z=0.8)
    assert spec.kappa_y == pytest.approx(5/6)
    with pytest.raises(ValueError, match="apply to beam2_timoshenko"):
        schema.ElementSpec(type="beam2", formulation="full", area=1e-2,
                           iy=1e-4, iz=2e-4, j=3e-4, ay=8e-3)


def test_short_cantilever_matches_bending_plus_shear_in_both_planes():
    s = _solver(iy=1.3e-4, iz=2.1e-4, ay=7e-3, az=4e-3,
                kappa_y=0.82, kappa_z=0.71)
    E = s.material.E; G = E/(2*(1+s.material.nu)); L = s.length[0]
    f = np.zeros(s.ndof); f[8] = -1200.0       # global z == local y
    q = _cantilever_linear_solution(s, f)
    expected = -1200*(L**3/(3*E*2.1e-4) + L/(0.82*G*7e-3))
    assert q[8] == pytest.approx(expected, rel=2e-13)
    f[:] = 0; f[7] = 900.0                     # global y == -local z
    q = _cantilever_linear_solution(s, f)
    expected = 900*(L**3/(3*E*1.3e-4) + L/(0.71*G*4e-3))
    assert q[7] == pytest.approx(expected, rel=2e-13)


def test_pure_axial_torsion_and_bending_are_not_changed_by_shear_flexibility():
    s = _solver()
    E = s.material.E; G = E/(2*(1+s.material.nu)); L = s.length[0]
    f = np.zeros(s.ndof); f[6] = 1000; f[9] = 500; f[11] = 200
    q = _cantilever_linear_solution(s, f)
    assert q[6] == pytest.approx(1000*L/(E*1e-2))
    assert q[9] == pytest.approx(500*L/(G*2e-4))
    assert q[11] == pytest.approx(200*L/(E*1e-4))
    assert q[7] == pytest.approx(200*L**2/(2*E*1e-4))


def test_slender_limit_converges_to_euler_bernoulli_matrix():
    timo = _solver(ay=1e12, az=1e12, kappa_y=1.0, kappa_z=1.0)
    case, _ = schema.load(str(BASE)); mesh, material, _ = build(case, str(BASE))
    eb = make_solver(_with_element(case, "beam2", str(BASE)), mesh, material)
    assert np.allclose(timo.k_local, eb.k_local, rtol=2e-14, atol=1e-3)


def test_corotational_timoshenko_beam_rejects_rigid_motion():
    s = _solver()
    angle = np.pi/3
    Q = np.array([[np.cos(angle), -np.sin(angle), 0.],
                  [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    q = s.q.reshape(-1, 6)
    q[:, :3] = s.mesh.nodes@Q.T-s.mesh.nodes+[0.2, -0.1, 0.4]
    q[:, 3:] = [0., 0., angle]
    s.sync_orientation()
    force, actions = s.internal_force()
    assert np.abs(force).max() < 1e-7
    assert np.abs(actions).max() < 1e-7


def test_timoshenko_release_has_zero_released_end_moment():
    s = _solver(releases=["j_rz"])
    # Any compatible deformation must leave the statically condensed local
    # end moment exactly zero; the released nodal rotation itself is inactive.
    s.q.reshape(-1, 6)[1, 1] = 1e-3
    s.internal_force()
    assert s.end_actions[0, 11] == pytest.approx(0.0, abs=1e-8)


def test_lumped_mass_reproduces_one_element_axial_and_torsional_frequencies():
    s = _solver()
    # One fixed-free element.  Translational and sectional rotary inertia are
    # deliberately checked independently of the time integrator.
    free = np.arange(6, 12)
    K = s.K[np.ix_(free, free)].toarray()
    M = np.diag(s.physical_mass[free])
    omega2 = eigh(K, M, eigvals_only=True)
    E=s.material.E; G=E/(2*(1+s.material.nu)); rho=s.material.rho; L=s.length[0]
    targets = [2*E/(rho*L**2), 2*G/(rho*L**2)]
    assert any(v == pytest.approx(targets[0], rel=1e-12) for v in omega2)
    assert any(v == pytest.approx(targets[1], rel=1e-12) for v in omega2)


def test_timoshenko_member_uses_common_mixed_line_assembly():
    case, _ = schema.load(str(ROOT/"cases"/"case_03_mixed_frame.yaml"))
    mesh, material, element = build(case, str(ROOT/"cases"/"case_03_mixed_frame.yaml"))
    p = _with_element(case, element)
    p["members"][0].update(type="beam2_timoshenko", ay=8e-3, az=8e-3,
                           kappa_y=5/6, kappa_z=5/6)
    s = make_solver(p, mesh, material)
    assert s.member_kind.tolist() == ["beam2_timoshenko", "truss2"]
    actions = s.internal_force()[1]
    assert s.nodal_fields(actions)["beam_formulation"].tolist() == [1., 0.]
