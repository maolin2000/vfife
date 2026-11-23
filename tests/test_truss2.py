"""Independent Line2 truss import and analytical axial-bar regression."""

from pathlib import Path

import numpy as np
import pytest

from vfife.cli import _with_element, build, make_solver
from vfife.io import gmsh, schema


ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT / "tests" / "data" / "bar_truss2_v22.msh"
CASE = ROOT / "cases" / "case_02_truss_bar.yaml"


def test_gmsh_imports_line2_and_physical_point_regions():
    mesh, element = gmsh.read(str(MESH))
    assert element == "truss2"
    assert mesh.elements.tolist() == [[0, 1]]
    assert mesh.regions["fixed"].tolist() == [0]
    assert mesh.regions["loaded"].tolist() == [1]


def test_standalone_bar_matches_pl_over_ea():
    case, _raw = schema.load(str(CASE))
    mesh, material, element = build(case, str(CASE))
    solver = make_solver(_with_element(case, element), mesh, material)
    solver.run(output_every=500, progress_every=500)
    # 2e-7 after the 500 iterations: kinetic damping that resets at the
    # kinetic-energy peaks. Until v1.0 it reset on EVERY step (the reset
    # kept the pre-reset energy as the one to beat), which on one degree of
    # freedom is a steepest descent and landed on 1e-9 -- and on a real model
    # was 48 times slower
    assert solver.u[1, 0] == pytest.approx(1000.0 / (2e11 * 1e-4), rel=1e-6)
    assert solver.u[:, 1:].max() == pytest.approx(0.0, abs=1e-15)
    f_int, stress = solver._force(True)
    assert f_int[0, 0] == pytest.approx(-1000.0, rel=1e-6)
    assert stress[0, 0, 0] == pytest.approx(10e6, rel=1e-6)
    fields = solver.nodal_fields(stress)
    assert fields["axial_force"][0] == pytest.approx(1000.0, rel=1e-6)
    assert np.allclose(fields["von_mises"], [10e6])


def test_truss_schema_requires_area_and_full_formulation():
    with pytest.raises(ValueError, match="area"):
        schema.ElementSpec(type="truss2", formulation="full")
    with pytest.raises(ValueError, match="full only"):
        schema.ElementSpec(type="truss2", formulation="b-bar", area=1e-4)


def test_truss_gravity_is_consistent_bar_volume_load_and_time_amplitudes_work():
    case, _ = schema.load(str(CASE))
    p = _with_element(case, "truss2")
    p["loads"] = [{"region": "bar", "gravity": [0, 0, -10.0], "amplitude": "pulse"}]
    p["amplitudes"] = {"pulse": {"table": [[0, 0], [0.5, 1], [1, 0]]}}
    mesh, material, _ = build(case, str(CASE))
    s = make_solver(p, mesh, material)
    force = s.external_force(0.5)
    assert force.sum(axis=0)[2] == pytest.approx(-7850.0 * 1e-4 * 10.0)
    assert np.allclose(force[[0, 1], 2], -7850.0 * 1e-4 * 10.0 / 2.0)


def test_truss_carries_base_load_through_a_second_step():
    case, _ = schema.load(str(CASE))
    p = _with_element(case, "truss2")
    p["solver"] = {**p["solver"], "n_steps": 500}
    p["steps"] = [{"name": "dead"}, {"name": "service", "loads": [
        {"region": "loaded", "vector": [1000.0, 0, 0]}]}]
    mesh, material, _ = build(case, str(CASE))
    s = make_solver(p, mesh, material)
    s.run(output_every=500, progress_every=500)
    assert s.u[1, 0] == pytest.approx(1e-4, rel=1e-6)      # see the test above
    assert s.times == pytest.approx([1.0, 2.0])


def test_truss_dynamic_path_reaches_the_static_bar_solution_with_damping():
    case, _ = schema.load(str(CASE))
    p = _with_element(case, "truss2")
    p["solver"] = {**p["solver"], "mode": "dynamic", "total_time": 0.01,
                   "damping": {"type": "viscous", "coeff": 2000.0}}
    mesh, material, _ = build(case, str(CASE))
    s = make_solver(p, mesh, material)
    s.run(output_every=10000, progress_every=10000)
    assert s.times == pytest.approx([0.01])
    assert s.u[1, 0] == pytest.approx(5e-5, rel=2e-4)


def test_ground_motion_uses_physical_mass_not_relaxation_mass(tmp_path):
    record = tmp_path / "motion.dat"
    record.write_text("0 0\n1 2\n", encoding="utf-8")
    case, _ = schema.load(str(CASE))
    p = _with_element(case, "truss2")
    p["solver"] = {**p["solver"], "mode": "dynamic", "total_time": 1.0,
                   "damping": {"type": "none"}}
    p["loads"] = [{"ground_motion": {"path": str(record), "direction": [1, 0, 0]}}]
    mesh, material, _ = build(case, str(CASE))
    s = make_solver(p, mesh, material)
    got = s.external_force(0.5, physical_time=0.5)
    assert np.allclose(got, -s.physical_mass * [1.0, 0.0, 0.0])


def test_nodal_mass_enters_dynamic_mass_and_ground_inertia(tmp_path):
    record = tmp_path / "motion.dat"
    record.write_text("0 0\n1 2\n", encoding="utf-8")
    case, _ = schema.load(str(CASE))
    p = _with_element(case, "truss2")
    p["solver"] = {**p["solver"], "mode": "dynamic", "total_time": 1.0,
                   "damping": {"type": "none"}}
    p["nodal_masses"] = [{"region": "loaded", "mass": 100.0, "per_node": True}]
    p["loads"] = [{"ground_motion": {"path": str(record), "direction": [1, 0, 0]}}]
    mesh, material, _ = build(case, str(CASE))
    s = make_solver(p, mesh, material)
    bar_end_mass = material.rho * p["element"]["area"] / 2.0
    assert s.physical_mass[1, 0] == pytest.approx(bar_end_mass + 100.0)
    assert s.mass[1, 0] == pytest.approx(bar_end_mass + 100.0)
    assert s.external_force(0.5, 0.5)[1, 0] == pytest.approx(-(bar_end_mass + 100.0))


def test_truss_rayleigh_damping_has_exact_mass_and_initial_stiffness_terms():
    case, _ = schema.load(str(CASE))
    p = _with_element(case, "truss2")
    p["solver"] = {**p["solver"], "mode": "dynamic", "total_time": 0.01,
                   "damping": {"type": "rayleigh", "alpha": 2.0, "beta": 1e-4}}
    mesh, material, _ = build(case, str(CASE))
    s = make_solver(p, mesh, material)
    s.v[1, 0] = 1.0
    force = s._rayleigh_force()
    k = material.E * p["element"]["area"]
    assert force[0, 0] == pytest.approx(-1e-4 * k)
    assert force[1, 0] == pytest.approx(2.0 * s.mass[1, 0] + 1e-4 * k)


@pytest.mark.parametrize(("motion", "value", "factor"), [
    ("velocity", 0.2, lambda t: 0.2 * t),
    ("acceleration", 4.0, lambda t: 0.5 * 4.0 * t * t),
])
def test_dynamic_kinematic_velocity_and_acceleration_constraints(motion, value, factor):
    case, _ = schema.load(str(CASE))
    p = _with_element(case, "truss2")
    p["solver"] = {**p["solver"], "mode": "dynamic", "total_time": 0.01,
                   "damping": {"type": "none"}}
    p["constraints"] = [
        {"region": "fixed", "dofs": [1, 1, 1]},
        {"region": "loaded", "dofs": [1, 0, 0], "motion": motion,
         "value": value, "amplitude": "hold"},
    ]
    p["loads"] = []
    mesh, material, _ = build(case, str(CASE))
    s = make_solver(p, mesh, material)
    s.run(output_every=10000, progress_every=10000)
    elapsed = s.n_steps * s.dt
    assert s.u[1, 0] == pytest.approx(factor(elapsed), rel=2e-3)
