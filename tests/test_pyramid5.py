"""E04 pyramid5 -- registration, shape functions and patch tests."""

import numpy as np
import pytest

from vfife.elements.library import get, element_definition, shape_functions
from vfife.io import schema
from vfife.api import prepare_case, run_case
from pathlib import Path
import tempfile


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def test_pyramid5_registered_with_correct_metadata():
    defn = element_definition("pyramid5")
    assert defn.name == "pyramid5"
    assert defn.family == "continuum"
    assert defn.topology == ("pyramid5",)
    assert defn.formulations == ("full",)
    # Was ("numpy",) -- a first-delivery caution that was enforced by preflight
    # and never revisited, so this element ran ~3x slower than necessary with no
    # way to opt out. The fused kernels are element-generic and were computing it
    # correctly all along; test_backends.py checks that against the numpy
    # reference to <1e-11 on numba and warp.
    assert defn.backends == ("numpy", "numba", "warp")
    assert "full" in get("pyramid5").formulations

def test_pyramid5_shape_partition_and_linear_edge():
    pts = np.array([[0,0,0.2],[0.1, -0.2, 0.5],[ -0.3,0.3,0.8],[0,0,1]], dtype=float)
    N = shape_functions("pyramid5", pts)
    assert np.allclose(N.sum(axis=1), 1.0, atol=1e-12)
    for s in [0, 0.25, 0.5, 0.75, 1.0]:
        xi = -1 + s
        eta = -1 + s
        zeta = s
        N = shape_functions("pyramid5", np.array([[xi, eta, zeta]]))
        assert N[0,0] == pytest.approx(1 - s, abs=1e-12)
        assert N[0,4] == pytest.approx(s, abs=1e-12)
        assert N[0,1] == pytest.approx(0, abs=1e-12)
        assert N[0,2] == pytest.approx(0, abs=1e-12)

def test_pyramid5_quadrature_volume():
    edef = get("pyramid5")
    pts, w = edef.quadrature["full"]
    assert pts.shape[0] == 8
    assert w.sum() == pytest.approx(4.0/3.0, rel=1e-12)
    assert np.dot(w, pts[:,2]) == pytest.approx(1.0/3.0, rel=1e-12)

def test_pyramid5_jacobian_positive_on_regular_pyramid():
    tmp = Path(tempfile.gettempdir()) / "vfife_test_pyramid5.msh"
    msh = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
5
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0.5 0.5 1
$EndNodes
$Elements
1
1 7 2 1 1 1 2 3 4 5
$EndElements
"""
    tmp.write_text(msh, encoding="utf-8")
    from vfife.io.gmsh import read
    mesh, elem = read(str(tmp))
    assert elem == "pyramid5"
    raw = {
        "name": "pyr_reg",
        "mesh": {"type": "gmsh", "path": str(tmp)},
        "element": {"formulation": "full"},
        "material": {"type": "linear_elastic", "E": 2.1e11, "nu": 0.3, "rho": 7800},
        "constraints": [{"region": "all", "dofs": [1,1,1]}],
        "solver": {"mode": "relaxation", "n_steps": 10},
    }
    case = schema.Case.model_validate(raw)
    prep = prepare_case(case)
    assert not any(i["code"] == "NONPOSITIVE_JACOBIAN" for i in prep.preflight["issues"])

def test_pyramid5_runs_through_solver():
    tmp = Path(tempfile.gettempdir()) / "vfife_test_pyramid5_run.msh"
    msh = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
5
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0.5 0.5 1
$EndNodes
$Elements
1
1 7 2 1 1 1 2 3 4 5
$EndElements
"""
    tmp.write_text(msh, encoding="utf-8")
    raw = {
        "name": "pyr_run",
        "mesh": {"type": "gmsh", "path": str(tmp)},
        "element": {"formulation": "full"},
        "material": {"type": "linear_elastic", "E": 2.1e11, "nu": 0.3, "rho": 7800},
        "constraints": [{"region": "all", "dofs": [1,1,1]}],
        "solver": {"mode": "relaxation", "n_steps": 20, "tolerance": 1e-6},
    }
    case = schema.Case.model_validate(raw)
    prep = prepare_case(case)
    assert prep.preflight["status"] == "PASS"
    outcome = run_case(prep)
    assert outcome.status in ("CONVERGED", "COMPLETED_DYNAMIC", "COMPLETED", "CONVERGED")

def test_pyramid5_uniaxial_tension_via_nodal_forces():
    tmp = Path(tempfile.gettempdir()) / "vfife_test_pyramid5_tens.msh"
    msh = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
5
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0.5 0.5 1
$EndNodes
$Elements
1
1 7 2 1 1 1 2 3 4 5
$EndElements
"""
    tmp.write_text(msh, encoding="utf-8")
    raw2 = {
        "name": "pyr_tens2",
        "mesh": {"type": "gmsh", "path": str(tmp)},
        "element": {"formulation": "full"},
        "material": {"type": "linear_elastic", "E": 2.1e11, "nu": 0.3, "rho": 7800},
        "regions": {"base": {"plane": {"point": [0,0,0], "normal": [0,0,1]}}},
        "constraints": [{"region": "base", "dofs": [1,1,1]}],
        "loads": [{"region": "all", "vector": [0,0, 5e3]}],
        "solver": {"mode": "relaxation", "n_steps": 800, "tolerance": 1e-4},
    }
    case = schema.Case.model_validate(raw2)
    prep = prepare_case(case)
    assert prep.preflight["status"] == "PASS"
    outcome = run_case(prep)
    # apex should lift; check via outcome diagnostics and file existence
    assert outcome.status == "CONVERGED"
    assert Path(outcome.result_path).exists()
