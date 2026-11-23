from pathlib import Path
import numpy as np
import pytest

from vfife.cli import build, make_solver, _with_element
from vfife.io import gmsh, schema
from vfife.preflight import check
from vfife.io.writer import ResultWriter

ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT / "tests" / "data" / "coupled_hex_truss_v22.msh"


def problem():
    return schema.Case.model_validate({
        "name": "coupled_patch", "mesh": {"type": "gmsh", "path": str(MESH)},
        "element": {"type": "coupled", "formulation": "full"},
        "members": [{"region": "truss_members", "type": "truss2", "area": 1e-3}],
        "material": {"type": "linear_elastic", "E": 2e11, "nu": .3, "rho": 7850},
        "constraints": [
            {"region": "fixed", "dofs": [1, 1, 1, 0, 0, 0]},
            {"region": "loaded", "dofs": [0, 1, 1, 0, 0, 0]}],
        "loads": [{"region": "loaded", "vector": [1000, 0, 0]}],
        "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 20},
    })


def test_coupled_import_preserves_shared_numbering():
    mesh, kind = gmsh.read_coupled(str(MESH))
    assert kind == "coupled"
    assert set(mesh.cell_blocks) == {"hexa8", "truss2"}
    assert mesh.cell_blocks["truss2"].tolist() == [[0, 1]]
    assert mesh.cell_blocks["hexa8"][0, :2].tolist() == [0, 1]


def test_shared_translation_receives_solid_and_truss_force():
    case = problem(); mesh, material, element = build(case, str(MESH))
    solver = make_solver(_with_element(case, element, str(MESH)), mesh, material)
    solver.q[6] = 1e-6
    total, _ = solver.internal_force()
    line_only, _ = super(type(solver), solver)._member_forces()
    assert abs(total[6]) > abs(line_only[6])
    assert np.isfinite(total).all()
    fixed = solver.step_data(0)[0].reshape(-1, 6)
    assert fixed[2:, 3:].all()  # solid-only rotational slots are inactive


def test_coupled_preflight_and_two_para_view_topologies(tmp_path):
    case = problem(); mesh, material, element = build(case, str(MESH))
    report = check(case, mesh, element)
    assert report["status"] == "PASS", report["issues"]
    solver = make_solver(_with_element(case, element, str(MESH)), mesh, material)
    solver.internal_force()
    path = tmp_path / "coupled.h5"
    writer = ResultWriter(str(path), mesh, element="coupled")
    writer.write_frame(1, 1.0, solver.nodal_fields(solver._cell_actions()), equilibrium=True)
    writer.close()
    assert path.with_suffix(".xdmf").exists()
    assert (tmp_path / "coupled_members.xdmf").exists()
    assert "/mesh/elements" in path.with_suffix(".xdmf").read_text(encoding="utf-8")
    assert "/members/elements" in (tmp_path / "coupled_members.xdmf").read_text(encoding="utf-8")


def test_coupled_dynamic_uses_combined_mass_and_completes():
    case = problem()
    case.solver.mode = "dynamic"; case.solver.total_time = 2e-5
    case.solver.damping.type = "none"; case.loads = []
    mesh, material, element = build(case, str(MESH))
    solver = make_solver(_with_element(case, element, str(MESH)), mesh, material)
    line_mass = material.rho * 1e-3 / 2
    assert solver.physical_mass.reshape(-1, 6)[0, 0] > line_mass
    solver.q[6] = 1e-8
    solver.run(output_every=999999, progress_every=999999)
    assert solver.diagnostics["status"] == "COMPLETED_DYNAMIC"
    assert np.isfinite(solver.q).all()


def test_coupled_beam_rotation_mechanism_is_not_hidden_by_tiny_mass():
    case = problem()
    case.members[0] = schema.MemberGroupSpec(
        region="truss_members", type="beam2", area=1e-3,
        iy=1e-5, iz=1e-5, j=2e-5, orientation=[0, 0, 1])
    mesh, material, element = build(case, str(MESH))
    report = check(case, mesh, element)
    assert any(x["code"] == "BEAM_ROTATION_MECHANISM" for x in report["issues"])
    with pytest.raises(ValueError, match="mechanism DOF"):
        make_solver(_with_element(case, element, str(MESH)), mesh, material)
    case.constraints[0].dofs = [True]*6
    solver = make_solver(_with_element(case, element, str(MESH)), mesh, material)
    assert solver.active_rot[0].all()


# --------------------------------------------------------------------------- #
#  loads a coupled model used to drop on the floor
# --------------------------------------------------------------------------- #
#  The solid half and the line half each took the load kinds they knew, by two
#  INDEPENDENT whitelists. A kind in neither -- `temperature`, `moving_load` --
#  was removed from both and the model ran, converged and reported success with
#  the load simply absent. Two whitelists cannot be audited against each other;
#  one partition with a remainder can.
def test_a_load_neither_half_owns_is_refused_by_name():
    case = problem()
    case.material["thermal_expansion"] = 1.2e-5
    case.loads = case.loads + [schema.LoadSpec.model_validate(
        {"region": "all", "temperature": {"uniform": 50.0}})]
    mesh, material, element = build(case, str(MESH))
    with pytest.raises(ValueError, match="does not carry a 'temperature' load"):
        make_solver(_with_element(case, element, str(MESH)), mesh, material)


def test_a_surface_load_written_inside_a_step_is_refused_not_dropped():
    """The solid half has no steps of its own -- it is evaluated at the
    parent's state -- so a step-level traction reached neither half."""
    case = problem()
    case.steps = [schema.StepSpec.model_validate(
        {"name": "press", "loads": [{"region": "loaded", "pressure": 1.0e5}]})]
    mesh, material, element = build(case, str(MESH))
    with pytest.raises(ValueError, match="no step of its own"):
        make_solver(_with_element(case, element, str(MESH)), mesh, material)


def test_the_loads_a_coupled_model_does_own_still_reach_their_half():
    """The partition is a partition: nothing that used to arrive stopped."""
    case = problem()
    case.loads = case.loads + [schema.LoadSpec.model_validate(
        {"region": "all", "pressure": 1.0e5})]
    mesh, material, element = build(case, str(MESH))
    solver = make_solver(_with_element(case, element, str(MESH)), mesh, material)
    assert [set(k for k in ("vector", "pressure") if x.get(k) is not None)
            for x in solver.solid.p["loads"]] == [{"pressure"}]
    assert [set(k for k in ("vector", "pressure") if x.get(k) is not None)
            for x in solver.p["loads"]] == [{"vector"}]


def test_a_lumped_mass_is_added_once_and_not_twice():
    """Both halves used to read `nodal_masses` off their own copy of the
    problem, and line 74 sums the two physical masses: a lump arrived at
    exactly twice its weight, so every natural period came out sqrt(2) too
    long and the summary still printed the number the analyst wrote."""
    lump = 1.0e6
    totals = []
    for masses in ([], [{"region": "all", "mass": lump}]):
        case = problem()
        case.solver.mode = "dynamic"; case.solver.total_time = 1e-6
        case.nodal_masses = [schema.NodalMassSpec.model_validate(m) for m in masses]
        mesh, material, element = build(case, str(MESH))
        solver = make_solver(_with_element(case, element, str(MESH)), mesh, material)
        totals.append(solver.physical_mass.reshape(-1, 6)[:, 0].sum())
    assert (totals[1] - totals[0]) == pytest.approx(lump, rel=1e-12)
    # and the solid half carries none of it, so it cannot come back
    case = problem(); case.nodal_masses = [schema.NodalMassSpec.model_validate(
        {"region": "all", "mass": lump})]
    mesh, material, element = build(case, str(MESH))
    solver = make_solver(_with_element(case, element, str(MESH)), mesh, material)
    assert not solver.solid.p["nodal_masses"]
    assert np.allclose(solver.solid.additional_mass, 0.0)
