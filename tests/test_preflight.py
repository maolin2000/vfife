"""Engineering model preflight checks."""
from pathlib import Path

import numpy as np

from vfife.cli import build
from vfife.core.mesh import Mesh
from vfife.io import schema
from vfife.preflight import check

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = ROOT / "cases" / name
    case, _ = schema.load(path)
    mesh, _material, element = build(case, str(path))
    return case, mesh, element


def _codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_all_numbered_cases_pass_preflight_without_initializing_cuda():
    for path in sorted((ROOT / "cases").glob("case_*.yaml")):
        case, _ = schema.load(path); mesh, _, element = build(case, str(path))
        assert check(case, mesh, element)["status"] == "PASS", path.name


def test_duplicate_element_orphan_and_zero_length_are_errors():
    case, mesh, element = _load("case_02_truss_bar.yaml")
    bad = Mesh(np.vstack([mesh.nodes, [[2., 0., 0.]]]),
               np.vstack([mesh.elements, mesh.elements]), dict(mesh.regions))
    report = check(case, bad, element)
    assert {"DUPLICATE_ELEMENT", "ORPHAN_NODE"} <= _codes(report)
    zero = Mesh(mesh.nodes.copy(), np.array([[0, 0]], dtype=np.int32), dict(mesh.regions))
    assert {"REPEATED_ELEMENT_NODE", "ZERO_LENGTH"} <= _codes(check(case, zero, element))


def test_missing_support_and_truss_mechanism_are_reported():
    case, mesh, element = _load("case_02_truss_bar.yaml")
    unsupported = case.model_copy(deep=True); unsupported.constraints = []
    codes = _codes(check(unsupported, mesh, element))
    assert "RIGID_BODY_MODE" in codes and "TRUSS_MECHANISM" in codes


def test_mixed_assignment_and_parallel_beam_orientation_are_errors():
    case, mesh, element = _load("case_03_mixed_frame.yaml")
    missing = case.model_copy(deep=True); missing.members = missing.members[:1]
    assert "UNASSIGNED_MEMBER" in _codes(check(missing, mesh, element))
    parallel = case.model_copy(deep=True); parallel.members[0].orientation = [1., 0., 0.]
    assert "BEAM_ORIENTATION" in _codes(check(parallel, mesh, element))


def test_dynamic_model_without_mass_is_an_error():
    case, mesh, element = _load("case_05_space_truss_seismic.yaml")
    massless = case.model_copy(deep=True); massless.material["rho"] = 0.; massless.nodal_masses = []
    assert "ZERO_DYNAMIC_MASS" in _codes(check(massless, mesh, element))


def test_floating_disconnected_beam_component_is_not_hidden_by_supported_part():
    case, mesh, element = _load("case_01_beam_cantilever.yaml")
    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [1., 2., 0.]])
    conn = np.array([[0, 1], [2, 3]], dtype=np.int32)
    disconnected = Mesh(nodes, conn, {"fixed": np.array([0]), "loaded": np.array([1])})
    report = check(case, disconnected, element)
    assert "RIGID_BODY_MODE" in _codes(report)


def test_a_free_rigid_mode_is_named_and_a_fix_is_proposed_as_yaml():
    """"5 of 6" left the analyst to find the sixth. case_12 shipped with a
    free rotation about z for a whole round of work (the support pin fixed x
    and y at one point, which cannot hold a rotation about the vertical), and
    the sixth constraint was found by trial. The message now names the mode
    from the null space of the restrained modes and proposes the region and
    dofs that remove it, at the node where that mode moves most."""
    case, mesh, element = _load("case_12_ualberta_sf0b.yaml")
    loose = case.model_copy(deep=True)
    loose.constraints = [c for c in loose.constraints if c.region != "pin_R"]
    report = check(loose, mesh, element)
    issue = next(i for i in report["issues"] if i["code"] == "RIGID_BODY_MODE")
    assert issue["details"]["unconstrained_modes"] == 1
    assert "rotation about z" in issue["message"]
    assert "where: {x:" in issue["message"] and "dofs: [0, 1, 0]" in issue["message"], issue["message"]
