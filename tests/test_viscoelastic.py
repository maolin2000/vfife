"""viscoelastic -- Prony shear, dynamic only."""
import numpy as np, tempfile
from pathlib import Path
import pytest
from vfife.materials import create, definition
from vfife.io import schema
from vfife.api import prepare_case, run_case


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def test_viscoelastic_registered():
    assert "viscoelastic" in [d.name for d in [definition(n) for n in ["viscoelastic"]]]
    assert definition("viscoelastic").kinematics == ("solid_3d",)
    assert definition("viscoelastic").backends == ("numpy",)

def test_viscoelastic_elastic_limit():
    m = create("viscoelastic", E=2.1e11, nu=0.3, rho=7850, g=[], tau=[])
    strain = np.array([[1e-3,0,0,0,0,0]], float)
    state = m.init_state(1)
    s = m.update(strain, state, dt=0.01)
    m2 = create("linear_elastic", E=2.1e11, nu=0.3, rho=7850)
    s2 = m2.update(strain, np.zeros((1,0)), dt=0.01)
    assert np.allclose(s, s2, atol=1e-6)

def test_viscoelastic_relaxation():
    m = create("viscoelastic", E=3e9, nu=0.35, rho=1200, g=[0.5], tau=[0.1])
    strain = np.array([[0.01,0,0,0,0,0]], float)
    state = m.init_state(1)
    s0 = m.update(strain, state.copy(), dt=1e-6)  # near instantaneous
    # hold strain constant for many steps
    s = s0
    for i in range(100):
        s = m.update(strain, state, dt=0.01)
    # stress should relax towards G_inf (half)
    assert s[0,0] < s0[0,0]
    assert s[0,0] > 0.4*s0[0,0]  # not fully relaxed

def _msh_hex8(tmp_path):
    # unit cube hexa8
    coords = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    nodes = "\n".join(f"{i+1} {x} {y} {z}" for i,(x,y,z) in enumerate(coords))
    elem = "1 5 2 1 1 " + " ".join(str(i+1) for i in range(8))
    msh = f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n8\n{nodes}\n$EndNodes\n$Elements\n1\n{elem}\n$EndElements\n"
    p = tmp_path/"hex.msh"; p.write_text(msh, encoding="utf-8"); return p

def test_viscoelastic_runs_dynamic(tmp_path):
    msh = _msh_hex8(tmp_path)
    raw={"name":"visco","mesh":{"type":"gmsh","path":str(msh)},"element":{"type":"hexa8","formulation":"full"},"material":{"type":"viscoelastic","E":3e9,"nu":0.35,"rho":1200,"g":[0.3],"tau":[0.01]},"solver":{"mode":"dynamic","backend":"numpy","total_time":0.001,"dt_safety":0.6},"constraints":[{"region":"base","dofs":[True,True,True]}],"loads":[{"region":"top","vector":[0,0,1000]}],"regions":{"base":{"plane":{"point":[0,0,0],"normal":[0,0,1]}},"top":{"plane":{"point":[0,0,1],"normal":[0,0,1]}}}}
    case=schema.Case.model_validate(raw)
    prep=prepare_case(case)
    assert prep.preflight["status"]=="PASS"
    out=run_case(prep)
    assert Path(out.result_path).exists()
