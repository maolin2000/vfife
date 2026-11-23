"""RC hybrid fiber beam: concrete01 + rebar."""
import tempfile, pathlib
from vfife.api import prepare_case, run_case
import pytest


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def _run(P, with_rebar=True, area=0.001):
    msh = pathlib.Path(tempfile.gettempdir()) / f"rc_{P}_{with_rebar}.msh"
    # use unique file per call to avoid cache
    import uuid
    msh = pathlib.Path(tempfile.gettempdir()) / f"rc_{uuid.uuid4().hex}.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n2\n1 0 0 0\n2 1 0 0\n$EndNodes\n$Elements\n1\n1 1 2 1 1 1 2\n$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(tempfile.gettempdir()) / f"rc_{uuid.uuid4().hex}.yaml"
    bars = ""
    if with_rebar:
        bars = "\n    bars: [{y: 0.0, z: 0.12, section: bar}, {y: 0.0, z: -0.12, section: bar}]"
    yp.write_text(f"""materials:
  concrete: {{type: concrete02, fpc: 30.0e6, epsc0: 0.002, fpcu: 6.0e6, epscu: 0.0035, rho: 2400, ft: 3.0e6, Ets: 1.5e9}}
  steel: {{type: rebar, E: 200.0e9, yield_strength: 400.0e6, hardening: 1.0e9, rho: 7850}}
sections:
  bar: {{type: bar, material: steel, shape: general, area: {area}}}
  rc:
    type: fiber
    material: concrete
    shape: rect
    width: 0.2
    height: 0.3
    ny: 8
    nz: 8{bars}
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
    vector: [0, 0, {P}]
solver:
  backend: numpy
  mode: relaxation
  n_steps: 1500
  load_steps: 20
output:
  monitors:
    tip_disp: {{region: tip, dof: z, kind: displacement}}
""", encoding="utf-8")
    prep = prepare_case(str(yp))
    assert prep.preflight["status"] == "PASS"
    out = run_case(prep)
    assert out.converged
    return float(out.monitors["tip_disp"][-1])

def test_rc_hybrid_runs():
    tip = _run(10000, with_rebar=True)
    assert tip > 0

def test_rc_stronger_than_plain():
    plain = _run(12000, with_rebar=False)
    rc = _run(12000, with_rebar=True, area=0.001)
    # RC should be stiffer (smaller tip) due to steel
    assert rc < plain
    assert plain - rc > 1e-06  # measurable

def test_rc_preflight_hybrid():
    from vfife.io.schema import Case
    c = Case.model_validate({
        "material": {"type": "concrete01", "E": 30e9, "fc": 30e6, "rho": 2400},
        "element": {"type": "beam2_fiber", "formulation": "full",
                    "fiber_section": {"type": "rect", "width": 0.2, "height": 0.3, "ny": 4, "nz": 4},
                    "fiber_rebars": [{"y": 0, "z": 0.1, "area": 0.0005,
                                      "material": {"type": "rebar", "E": 200e9, "yield_strength": 400e6, "rho": 7850}}]},
        "mesh": {"type": "box", "dimensions": [1,1,1], "divisions": [1,1,1]},
        "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 10}
    })
    assert c.element.fiber_rebars[0]["area"] == 0.0005
