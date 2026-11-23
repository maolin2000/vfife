"""E04 hexa20 -- 20-node serendipity hexahedron."""
import numpy as np
import pytest
from pathlib import Path
import tempfile

from vfife.elements.library import get, element_definition, shape_functions
from vfife.io import schema
from vfife.api import prepare_case, run_case


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def test_hexa20_registered_with_correct_metadata():
    defn = element_definition("hexa20")
    assert defn.name == "hexa20"
    assert defn.family == "continuum"
    assert defn.topology == ("hexa20",)
    assert defn.formulations == ("full",)
    # Was ("numpy",) -- a first-delivery caution that was enforced by preflight
    # and never revisited, so this element ran ~3x slower than necessary with no
    # way to opt out. The fused kernels are element-generic and were computing it
    # correctly all along; test_backends.py checks that against the numpy
    # reference to <1e-11 on numba and warp.
    assert defn.backends == ("numpy", "numba", "warp")
    assert "full" in get("hexa20").formulations

def test_hexa20_shape_partition_and_corner():
    pts = np.array([[0,0,0], [0.5,-0.3,0.2], [-0.9,0.9,-0.8], [1,1,1], [-1,-1,-1]], dtype=float)
    N = shape_functions("hexa20", pts)
    assert np.allclose(N.sum(axis=1), 1.0, atol=1e-12)
    # corner node 0 at (-1,-1,-1) should be 1 at that point
    N0 = shape_functions("hexa20", np.array([[-1,-1,-1]], dtype=float))
    assert N0[0,0] == pytest.approx(1.0, abs=1e-12)
    assert N0[0,1:].sum() == pytest.approx(0.0, abs=1e-12)
    # mids: edge 8 at (0,-1,-1) should be 1 at that point
    Nmid = shape_functions("hexa20", np.array([[0,-1,-1]], dtype=float))
    assert Nmid[0,8] == pytest.approx(1.0, abs=1e-12)

def test_hexa20_quadrature_volume():
    edef = get("hexa20")
    pts, w = edef.quadrature["full"]
    assert pts.shape == (27, 3)
    assert w.sum() == pytest.approx(8.0, rel=1e-12)
    # linear field integral: sum w * xi should be 0
    assert np.dot(w, pts[:,0]) == pytest.approx(0.0, abs=1e-12)

def test_hexa20_jacobian_positive_on_unit_cube():
    tmp = Path(tempfile.gettempdir()) / "vfife_test_hexa20_jac.msh"
    nodes_txt="\n".join(f"{i+1} {x} {y} {z}" for i,(x,y,z) in enumerate([
        # NOTE: Gmsh's own edge order, which is NOT VTK's. These fixtures used to
        # list the mid-side nodes in VTK order inside a file labelled Gmsh type
        # 17/18 -- a file Gmsh never writes. The reader's permutation was the
        # identity, so the two errors cancelled and the tests passed while no
        # real Gmsh hexa20 or wedge15 file could be read at all.
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1],
        [0.5,0,0],[0,0.5,0],[0,0,0.5],[1,0.5,0],[1,0,0.5],[0.5,1,0],
        [1,1,0.5],[0,1,0.5],[0.5,0,1],[0,0.5,1],[1,0.5,1],[0.5,1,1]]))
    conn=" ".join(str(i+1) for i in range(20))
    msh=f"""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
20
{nodes_txt}
$EndNodes
$Elements
1
1 17 2 1 1 {conn}
$EndElements
"""
    tmp.write_text(msh, encoding="utf-8")
    from vfife.io.gmsh import read
    mesh, elem = read(str(tmp))
    assert elem == "hexa20"
    raw = {"name":"hex20_jac","mesh":{"type":"gmsh","path":str(tmp)},"element":{"formulation":"full"},"material":{"type":"linear_elastic","E":2.1e11,"nu":0.3,"rho":7800},"constraints":[{"region":"all","dofs":[1,1,1]}],"solver":{"mode":"relaxation","n_steps":10}}
    case = schema.Case.model_validate(raw)
    prep = prepare_case(case)
    assert not any(i["code"] == "NONPOSITIVE_JACOBIAN" for i in prep.preflight["issues"])

def test_hexa20_runs_through_solver():
    tmp = Path(tempfile.gettempdir()) / "vfife_test_hexa20_run.msh"
    nodes_txt="\n".join(f"{i+1} {x} {y} {z}" for i,(x,y,z) in enumerate([
        # NOTE: Gmsh's own edge order, which is NOT VTK's. These fixtures used to
        # list the mid-side nodes in VTK order inside a file labelled Gmsh type
        # 17/18 -- a file Gmsh never writes. The reader's permutation was the
        # identity, so the two errors cancelled and the tests passed while no
        # real Gmsh hexa20 or wedge15 file could be read at all.
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1],
        [0.5,0,0],[0,0.5,0],[0,0,0.5],[1,0.5,0],[1,0,0.5],[0.5,1,0],
        [1,1,0.5],[0,1,0.5],[0.5,0,1],[0,0.5,1],[1,0.5,1],[0.5,1,1]]))
    conn=" ".join(str(i+1) for i in range(20))
    msh=f"""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
20
{nodes_txt}
$EndNodes
$Elements
1
1 17 2 1 1 {conn}
$EndElements
"""
    tmp.write_text(msh, encoding="utf-8")
    raw = {"name":"hex20_run","mesh":{"type":"gmsh","path":str(tmp)},"element":{"formulation":"full"},"material":{"type":"linear_elastic","E":2.1e11,"nu":0.3,"rho":7800},"constraints":[{"region":"all","dofs":[1,1,1]}],"solver":{"mode":"relaxation","n_steps":20,"tolerance":1e-6}}
    case = schema.Case.model_validate(raw)
    prep = prepare_case(case)
    assert prep.preflight["status"] == "PASS"
    outcome = run_case(prep)
    assert outcome.status in ("CONVERGED","COMPLETED","COMPLETED_DYNAMIC")

def test_hexa20_uniaxial_tension_via_nodal_forces():
    tmp = Path(tempfile.gettempdir()) / "vfife_test_hexa20_tens.msh"
    nodes_txt="\n".join(f"{i+1} {x} {y} {z}" for i,(x,y,z) in enumerate([
        # NOTE: Gmsh's own edge order, which is NOT VTK's. These fixtures used to
        # list the mid-side nodes in VTK order inside a file labelled Gmsh type
        # 17/18 -- a file Gmsh never writes. The reader's permutation was the
        # identity, so the two errors cancelled and the tests passed while no
        # real Gmsh hexa20 or wedge15 file could be read at all.
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1],
        [0.5,0,0],[0,0.5,0],[0,0,0.5],[1,0.5,0],[1,0,0.5],[0.5,1,0],
        [1,1,0.5],[0,1,0.5],[0.5,0,1],[0,0.5,1],[1,0.5,1],[0.5,1,1]]))
    conn=" ".join(str(i+1) for i in range(20))
    msh=f"""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
20
{nodes_txt}
$EndNodes
$Elements
1
1 17 2 1 1 {conn}
$EndElements
"""
    tmp.write_text(msh, encoding="utf-8")
    raw2 = {"name":"hex20_tens","mesh":{"type":"gmsh","path":str(tmp)},"element":{"formulation":"full"},"material":{"type":"linear_elastic","E":2.1e11,"nu":0.3,"rho":7800},"regions":{"base":{"plane":{"point":[0,0,0],"normal":[0,0,1]}}},"constraints":[{"region":"base","dofs":[1,1,1]}],"loads":[{"region":"all","vector":[0,0,5e3]}],"solver":{"mode":"relaxation","n_steps":800,"tolerance":1e-4}}
    case = schema.Case.model_validate(raw2)
    prep = prepare_case(case)
    assert prep.preflight["status"] == "PASS"
    outcome = run_case(prep)
    assert outcome.status == "CONVERGED"
    assert Path(outcome.result_path).exists()
