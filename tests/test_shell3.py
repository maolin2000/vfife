"""shell3 -- 3-node flat Tri3 Mindlin shell."""
import pytest
import tempfile
from pathlib import Path
import numpy as np
from vfife.elements.library import element_definition
from vfife.elements import shell
from vfife.io import schema
from vfife.api import prepare_case, run_case


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def test_shell3_registered():
    defn = element_definition("shell3")
    assert defn.name == "shell3"
    assert defn.family == "surface"
    assert defn.topology == ("tri3",)
    assert "mindlin_reissner_3d" in defn.formulations

def test_shell3_shape_partition():
    N,d = shell.shape_tri(0.2,0.3)
    assert N.sum() == pytest.approx(1.0)
    assert d.shape == (2,3)
    for xi,eta in [(0,0),(1,0),(0,1),(1/3,1/3)]:
        N,_=shell.shape_tri(xi,eta)
        assert N.sum() == pytest.approx(1.0)

def test_shell3_frame_and_area():
    nodes = np.array([[0,0,0],[1,0,0],[0,1,0]], float)
    R,c,local = shell.frame_tri(nodes)
    assert local[2,2]==0
    data = shell.precompute_tri(nodes, 2.1e11,0.3,0.01)
    assert data.area == pytest.approx(0.5, rel=1e-12)
    assert data.stiffness.shape == (18,18)

def test_shell3_jacobian_positive():
    tmp = Path(tempfile.gettempdir()) / "vfife_test_shell3_jac.msh"
    msh="""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0 0 0
2 1 0 0
3 0 1 0
$EndNodes
$Elements
1
1 2 2 1 1 1 2 3
$EndElements
"""
    tmp.write_text(msh, encoding="utf-8")
    from vfife.io.gmsh import read
    mesh,elem = read(str(tmp))
    assert elem=="shell3"
    raw={"name":"shell3_jac","mesh":{"type":"gmsh","path":str(tmp)},"element":{"type":"shell3","formulation":"selective","thickness":0.01},"material":{"type":"linear_elastic","E":2.1e11,"nu":0.3,"rho":7800},"constraints":[{"region":"all","dofs":[1,1,1,1,1,1]}],"solver":{"mode":"relaxation","n_steps":10}}
    case=schema.Case.model_validate(raw)
    prep=prepare_case(case)
    assert not any(i["code"]=="INVALID_SHELL_FACE" for i in prep.preflight["issues"])

def test_shell3_runs_through_solver():
    tmp = Path(tempfile.gettempdir()) / "vfife_test_shell3_run.msh"
    msh="""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0 0 0
2 1 0 0
3 0 1 0
$EndNodes
$Elements
1
1 2 2 1 1 1 2 3
$EndElements
"""
    tmp.write_text(msh, encoding="utf-8")
    raw={"name":"shell3_run","mesh":{"type":"gmsh","path":str(tmp)},"element":{"type":"shell3","formulation":"selective","thickness":0.01},"material":{"type":"linear_elastic","E":2.1e11,"nu":0.3,"rho":7800},"constraints":[{"region":"all","dofs":[1,1,1,1,1,1]}],"solver":{"mode":"relaxation","n_steps":20,"tolerance":1e-6}}
    case=schema.Case.model_validate(raw)
    prep=prepare_case(case)
    assert prep.preflight["status"]=="PASS"
    out=run_case(prep)
    assert out.status in ("CONVERGED","COMPLETED","COMPLETED_DYNAMIC")

def test_shell3_cantilever():
    tmp = Path(tempfile.gettempdir()) / "vfife_test_shell3_cant.msh"
    msh="""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
2
1 2 2 1 1 1 2 3
2 2 2 1 1 1 3 4
$EndElements
"""
    tmp.write_text(msh, encoding="utf-8")
    raw={"name":"shell3_cant","mesh":{"type":"gmsh","path":str(tmp)},"element":{"type":"shell3","formulation":"selective","thickness":0.01},"material":{"type":"linear_elastic","E":2.1e11,"nu":0.3,"rho":7800},"regions":{"fixed_edge":{"box":[-0.1,0.1,-0.1,1.1,-0.1,0.1]}},"constraints":[{"region":"fixed_edge","dofs":[1,1,1,1,1,1]}],"loads":[{"region":"all","vector":[0,0,100]}],"solver":{"mode":"relaxation","n_steps":800,"tolerance":1e-4}}
    case=schema.Case.model_validate(raw)
    prep=prepare_case(case)
    assert prep.preflight["status"]=="PASS"
    out=run_case(prep)
    # shell cantilever is a slow convergence case; allow MAX_STEPS as long as result exists
    assert out.status in ("CONVERGED","MAX_STEPS")
    assert Path(out.result_path).exists()
