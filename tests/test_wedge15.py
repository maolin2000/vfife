"""wedge15 -- 15-node quadratic prism."""
import tempfile
from pathlib import Path
import numpy as np
import pytest
from vfife.elements.library import element_definition, get, shape_functions
from vfife.io import schema
from vfife.api import prepare_case, run_case


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def test_wedge15_registered_with_correct_metadata():
    defn = element_definition("wedge15")
    assert defn.name == "wedge15"
    assert defn.family == "continuum"
    assert defn.topology == ("wedge15",)
    assert defn.formulations == ("full",)
    # Was ("numpy",) -- a first-delivery caution that was enforced by preflight
    # and never revisited, so this element ran ~3x slower than necessary with no
    # way to opt out. The fused kernels are element-generic and were computing it
    # correctly all along; test_backends.py checks that against the numpy
    # reference to <1e-11 on numba and warp.
    assert defn.backends == ("numpy", "numba", "warp")

def test_wedge15_shape_partition_and_corner():
    pt = np.array([[0,0,-1]], float)
    N = shape_functions("wedge15", pt)
    assert N.shape == (1,15)
    assert abs(N.sum() - 1) < 1e-12
    assert abs(N[0,0] - 1) < 1e-12

def test_wedge15_quadrature_volume():
    e = get("wedge15")
    pts, w = e.quadrature["full"]
    assert pts.shape[0] == 9
    assert abs(w.sum() - 1.0) < 1e-12
    # volume of unit wedge via integration
    xy = np.array([[0,0,-1],[1,0,-1],[0,1,-1],[0,0,1],[1,0,1],[0,1,1],
    [0.5,0,-1],[0.5,0.5,-1],[0,0.5,-1],[0.5,0,1],[0.5,0.5,1],[0,0.5,1],
    [0,0,0],[1,0,0],[0,1,0]], float)
    d = e.dN_dxi(pts)
    vol = sum(wi * np.linalg.det(d[k] @ xy) for k, wi in enumerate(w))
    assert abs(vol - 1.0) < 1e-8

def test_wedge15_jacobian_positive_on_unit_wedge():
    e = get("wedge15")
    xy = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,0,1],[0,1,1],
    [0.5,0,0],[0.5,0.5,0],[0,0.5,0],[0.5,0,1],[0.5,0.5,1],[0,0.5,1],
    [0,0,0.5],[1,0,0.5],[0,1,0.5]], float)
    pts, _ = e.quadrature["full"]
    d = e.dN_dxi(pts)
    for k in range(len(pts)):
        assert np.linalg.det(d[k] @ xy) > 0

def _msh_wedge15(tmp_path):
    tmp = tmp_path / "w15.msh"
    # Gmsh's own edge order, which is NOT the VTK order the solver uses
    # internally: (0,1),(0,2),(0,3),(1,2),(1,4),(2,5),(3,4),(3,5),(4,5).
    # This fixture used to hold the VTK order inside a file labelled Gmsh type
    # 18 -- a file Gmsh never writes -- and the reader's permutation was the
    # identity, so the two errors cancelled here while no real Gmsh 15-node
    # prism could be read at all.
    coords = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,0,1),(0,1,1),
     (0.5,0,0),(0,0.5,0),(0,0,0.5),(0.5,0.5,0),(1,0,0.5),
     (0,1,0.5),(0.5,0,1),(0,0.5,1),(0.5,0.5,1)]
    nodes = "\n".join(f"{i+1} {x} {y} {z}" for i,(x,y,z) in enumerate(coords))
    elem = "1 18 2 1 1 " + " ".join(str(i+1) for i in range(15))
    msh = f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n15\n{nodes}\n$EndNodes\n$Elements\n1\n{elem}\n$EndElements\n"
    tmp.write_text(msh, encoding="utf-8")
    return tmp

def test_wedge15_runs_through_solver(tmp_path):
    tmp = _msh_wedge15(tmp_path)
    raw={"name":"w15","mesh":{"type":"gmsh","path":str(tmp)},"element":{"type":"wedge15","formulation":"full"},"material":{"type":"linear_elastic","E":2.1e11,"nu":0.3,"rho":7850},"solver":{"mode":"relaxation","backend":"numpy","n_steps":800,"tolerance":1e-4},"constraints":[{"region":"all","dofs":[True,True,True]}],"loads":[],"regions":{"all":{"all":True}}}
    case=schema.Case.model_validate(raw)
    prep=prepare_case(case)
    assert prep.preflight["status"]=="PASS"
    out=run_case(prep)
    assert out.status=="CONVERGED"
    assert Path(out.result_path).exists()

def test_wedge15_uniaxial_tension_via_nodal_forces(tmp_path):
    tmp = _msh_wedge15(tmp_path)
    raw={"name":"w15_cant","mesh":{"type":"gmsh","path":str(tmp)},"element":{"type":"wedge15","formulation":"full"},"material":{"type":"linear_elastic","E":2.1e11,"nu":0.3,"rho":7850},"solver":{"mode":"relaxation","backend":"numpy","n_steps":800,"tolerance":1e-4},"constraints":[{"region":"base","dofs":[True,True,True]}],"loads":[{"region":"top","vector":[0,0,1e3]}],"regions":{"base":{"plane":{"point":[0,0,0],"normal":[0,0,1]}},"top":{"plane":{"point":[0,0,1],"normal":[0,0,1]}}}}
    case=schema.Case.model_validate(raw)
    prep=prepare_case(case)
    assert prep.preflight["status"]=="PASS"
    out=run_case(prep)
    assert Path(out.result_path).exists()
