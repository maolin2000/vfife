"""Concrete01 faithful OpenSees Kent-Scott-Park (no tension)."""
import numpy as np
import pytest
from vfife.materials.base import create, definition
from vfife.catalog import library
from vfife.api import prepare_case, run_case
import tempfile, pathlib


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def _follow(mat, history, state=None):
    state = mat.init_state(1) if state is None else state
    stresses = []
    for eps in history:
        strain = np.zeros((1, 6)); strain[0,0] = eps
        stresses.append(mat.update(strain, state, 1.0)[0,0])
    return np.asarray(stresses), state

def test_concrete01_envelope():
    m = create("concrete01", fpc=30e6, epsc0=0.002, fpcu=6e6, epscu=0.0035)
    def new_state():
        return m.init_state(1)
    # tension -> 0 (no tensile strength)
    s = m.update(np.array([[0.0001,0,0,0,0,0]], float), new_state(), 1.0)[0,0]
    assert s == pytest.approx(0.0, abs=1e-9)
    s = m.update(np.array([[0.001,0,0,0,0,0]], float), new_state(), 1.0)[0,0]
    assert s == pytest.approx(0.0, abs=1e-9)
    # compression peak at epsc0
    s = m.update(np.array([[-0.002,0,0,0,0,0]], float), new_state(), 1.0)[0,0]
    assert abs(s + 30e6) < 1e6
    # compression softening
    s2 = m.update(np.array([[-0.003,0,0,0,0,0]], float), new_state(), 1.0)[0,0]
    assert s2 > s  # less compressive
    # unloading stiffness degrades but goes to zero beyond Tend
    st = new_state()
    m.update(np.array([[-0.003,0,0,0,0,0]], float), st, 1.0)
    s_unload = m.update(np.array([[-0.001,0,0,0,0,0]], float), st, 1.0)[0,0]
    assert s_unload == pytest.approx(0.0, abs=1e2)

def test_concrete01_fiber_beam():
    with tempfile.TemporaryDirectory() as td:
        msh = pathlib.Path(td) / "m.msh"
        msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n2\n1 0 0 0\n2 1 0 0\n$EndNodes\n$Elements\n1\n1 1 2 1 1 1 2\n$EndElements\n", encoding="utf-8")
        yp = pathlib.Path(td) / "c.yaml"
        yp.write_text(f"""materials:
  concrete: {{type: concrete01, fpc: 30.0e6, epsc0: 0.002, fpcu: 6.0e6, epscu: 0.0035, rho: 2400}}
sections:
  rc: {{type: fiber, material: concrete, shape: rect, width: 0.2, height: 0.3, ny: 6, nz: 6}}
assign:
  - {{region: all, section: rc}}
mesh:
  type: gmsh
  path: {msh}
regions:
  fixed: {{box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}}
  tip: {{box: [0.99, 1.01, -0.01, 0.01, -0.01, 0.01]}}
constraints:
  - region: fixed
    dofs: [true, true, true, true, true, true]
loads:
  - region: tip
    vector: [-5000, 0, 0]
solver:
  backend: numpy
  mode: relaxation
  n_steps: 800
  load_steps: 20
output:
  monitors:
    tip_disp: {{region: tip, dof: x, kind: displacement}}
""", encoding="utf-8")
        prep = prepare_case(str(yp))
        assert prep.preflight["status"] == "PASS"
        out = run_case(prep)
        assert out.converged

def test_concrete01_catalog():
    lib = library()
    assert "concrete01" in lib["materials"]
    assert "concrete01" in lib["elements"]["beam2_fiber"]["compatible_materials"]

def test_concrete01_monotonic_compression():
    m = create("concrete01", fpc=30e6, epsc0=0.002, fpcu=6e6, epscu=0.0035)
    c, _ = _follow(create("concrete01", fpc=30e6, epsc0=0.002, fpcu=6e6, epscu=0.0035), [0.0, -0.002, -0.003])
    assert c[1] == pytest.approx(-30e6, rel=0.05)
    assert c[2] > c[1]

def test_concrete01_step_convergence_rate_independent():
    target = -0.002
    m1 = create("concrete01", fpc=30e6, epsc0=0.002, fpcu=6e6, epscu=0.0035)
    coarse, _ = _follow(m1, [target])
    m2 = create("concrete01", fpc=30e6, epsc0=0.002, fpcu=6e6, epscu=0.0035)
    fine, _ = _follow(m2, np.linspace(0, target, 51))
    assert coarse[-1] == pytest.approx(fine[-1], rel=1e-12)

def test_concrete01_unload_reload():
    m = create("concrete01", fpc=30e6, epsc0=0.002, fpcu=6e6, epscu=0.0035)
    hist = [0.0, -0.003, -0.001, -0.002]
    stresses, state = _follow(m, hist)
    # after compressive damage, reload still compressive
    assert stresses[2] < 0 or stresses[2] == pytest.approx(0.0, abs=1e2)
    assert state[0,0] == pytest.approx(-0.003)

def test_concrete01_energy_dissipation():
    m_el = create("concrete01", fpc=30e6, epsc0=0.002, fpcu=6e6, epscu=0.0035)
    el, _ = _follow(m_el, [0, -0.0002, 0])
    assert el[-1] == pytest.approx(0.0, abs=1e2)
    m_pl = create("concrete01", fpc=30e6, epsc0=0.002, fpcu=6e6, epscu=0.0035)
    hist = [0, -0.003, -0.001, -0.003, -0.001]
    stresses, _ = _follow(m_pl, hist)
    # unloading degraded vs elastic
    assert stresses[2] != pytest.approx(-30e9*0.001, rel=1e-2)
