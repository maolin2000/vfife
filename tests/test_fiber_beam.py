"""Fiber beam tests: axial fiber + elastic bending."""

import numpy as np
import pytest
from vfife.materials.base import create
from vfife.io.schema import Case, MeshSpec, ElementSpec
from vfife.cli import build
from vfife.api import prepare_case
import tempfile, os, pathlib


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def _case_yaml(tmpdir, mat_type="rebar", E=200e9, sy=400e6):
    msh = pathlib.Path(tmpdir) / "fiber.msh"
    msh.write_text("""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
2
1 0 0 0
2 1 0 0
$EndNodes
$Elements
1
1 1 2 1 1 1 2
$EndElements
""", encoding="utf-8")
    yaml_p = pathlib.Path(tmpdir) / "case.yaml"
    if mat_type == "rebar":
        mat_block = f"{{type: rebar, E: {E}, yield_strength: {sy}, hardening: 1.0e9, rho: 7850.0}}"
    else:
        mat_block = f"{{type: linear_elastic, E: {E}, nu: 0.3, rho: 7850.0}}"
    yaml_p.write_text(f"""materials:
  steel: {mat_block}
sections:
  rc: {{type: fiber, material: steel, shape: rect, width: 0.05, height: 0.10, ny: 6, nz: 6}}
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
    vector: [ 50000, 0, 0 ]
solver:
  backend: numpy
  mode: relaxation
  n_steps: 1200
  # 1e-3 on a 50 kN load is 50 N of force balance, and the assertion below is
  # on 2% of an elongation. 1e-4 cost 9680 internal-force evaluations per case
  # -- every stage ran to its budget, because a stiff axial DOF converges
  # slowly under relaxation -- and bought three digits nothing here reads.
  tolerance: 1.0e-3
  # An elastic axial pull does not need the 20 load stages a nonlinear material
  # defaults to.
  load_steps: 3
  max_steps_per_load: 1500
output:
  monitors:
    tip_disp: {{region: tip, dof: x, kind: displacement}}
""", encoding="utf-8")
    return str(yaml_p)

def test_fiber_beam_axial_elastic():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        yp = _case_yaml(td, mat_type="rebar", E=200e9, sy=1e12)
        from vfife.api import prepare_case
        prep = prepare_case(yp)
        assert prep.preflight["status"] == "PASS"
        # run
        from vfife.api import run_case
        out = run_case(prep)
        assert out.converged
        # axial displacement: P*L/(E*A) ; A=0.005, L=1, P=50000 => 50000/ (200e9*0.005)=5e-5
        tip = float(out.monitors["tip_disp"][-1])
        assert abs(tip - 5e-05) / 5e-05 < 0.05

def test_fiber_beam_axial_plastic():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Use high load to trigger yielding: stress = P/A = 600k/0.005=120 MPa >100
        yp = _case_yaml(td, mat_type="rebar", E=200e9, sy=100e6)
        # patch load to higher value
        import pathlib, yaml
        p = pathlib.Path(yp)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        data["loads"][0]["vector"] = [600000, 0, 0]
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        from vfife.api import prepare_case, run_case
        prep = prepare_case(yp)
        out = run_case(prep)
        assert out.converged
        tip = float(out.monitors["tip_disp"][-1])
        # elastic would be 600k/(200e9*0.005)=6e-4, plastic should be larger due to hardening 1e9
        assert tip > 8e-04

def test_fiber_beam_bending_matches_beam2():
    import tempfile, pathlib
    # compare fiber vs beam2 for same section/elastic
    with tempfile.TemporaryDirectory() as td:
        msh = pathlib.Path(td) / "b.msh"
        msh.write_text("""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
2
1 0 0 0
2 1 0 0
$EndNodes
$Elements
1
1 1 2 1 1 1 2
$EndElements
""", encoding="utf-8")
        for etype in ["beam2", "beam2_fiber"]:
            # beam2 needs area/I, fiber needs section
            if etype == "beam2":
                elem = "{type: beam, material: steel, shape: general, area: 0.005, iy: 4.166e-06, iz: 1.041e-06, j: 5.2e-06}"
                mat = "{type: linear_elastic, E: 200.0e9, nu: 0.3, rho: 7850.0}"
            else:
                elem = "{type: fiber, material: steel, shape: rect, width: 0.05, height: 0.10, ny: 10, nz: 10}"
                mat = "{type: rebar, E: 200.0e9, yield_strength: 1.0e12, hardening: 0, rho: 7850.0}"
            yp = pathlib.Path(td) / f"{etype}.yaml"
            yp.write_text(f"""materials:
  steel: {mat}
sections:
  sec: {elem}
assign:
  - {{region: all, section: sec}}
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
    vector: [0, 0, 100]
solver:
  backend: numpy
  mode: relaxation
  # 600 -> 3000 when the frame solvers gained the central-difference HALF-STEP
  # restart. The old full-step restart injected energy every time kinetic
  # damping fired, which converged faster and was wrong; this case stopped at
  # residual 9.5e-4 against 1e-4. The compared ANSWER is unchanged -- the
  # assertion below is fibre-vs-beam2 agreement, and both sides moved together.
  n_steps: 3000
output:
  monitors:
    tip_disp: {{region: tip, dof: z, kind: displacement}}
""", encoding="utf-8")
            from vfife.api import prepare_case, run_case
            prep = prepare_case(str(yp))
            out = run_case(prep)
            assert out.converged
            tip = float(out.monitors["tip_disp"][-1])
            # store
            if etype == "beam2":
                beam_tip = tip
            else:
                fiber_tip = tip
        assert abs(fiber_tip - beam_tip) / abs(beam_tip) < 0.08

def test_fiber_beam_preflight():
    from vfife.io.schema import Case
    c = Case.model_validate({"material":{"type":"rebar","E":200e9,"yield_strength":400e6,"rho":7850},
                              "element":{"type":"beam2_fiber","formulation":"full","fiber_section":{"type":"rect","width":0.1,"height":0.2,"ny":4,"nz":4}},
                              "mesh":{"type":"box","dimensions":[1,1,1],"divisions":[1,1,1]},
                              "solver":{"backend":"numpy","mode":"relaxation","n_steps":10}})
    assert c.element.type == "beam2_fiber"

def test_fiber_beam_bending_plastic():
    # low yield should soften bending response (larger tip than elastic)
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        msh = pathlib.Path(td) / "b.msh"
        msh.write_text("""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
2
1 0 0 0
2 1 0 0
$EndNodes
$Elements
1
1 1 2 1 1 1 2
$EndElements
""", encoding="utf-8")
        def run(sy):
            yp = pathlib.Path(td) / f"c{sy}.yaml"
            yp.write_text(f"""materials:
  steel: {{type: rebar, E: 200.0e9, yield_strength: {sy}, hardening: 1.0e9, rho: 7850.0}}
sections:
  # 2 x 2 on purpose: these tests check P*L/(EA) and the reported section
  # force, and A is the same however the section is diced. Finer fibres
  # cost solve time and prove nothing here.
  rc: {{type: fiber, material: steel, shape: rect, width: 0.05, height: 0.10, ny: 2, nz: 2}}
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
    vector: [0, 0, 1000]
solver:
  backend: numpy
  mode: relaxation
  n_steps: 800
output:
  monitors:
    tip_disp: {{region: tip, dof: z, kind: displacement}}
""", encoding="utf-8")
            from vfife.api import prepare_case, run_case
            prep = prepare_case(str(yp))
            out = run_case(prep)
            assert out.converged
            return float(out.monitors["tip_disp"][-1])
        elastic = run(1e12)
        plastic = run(10e6)
        assert plastic > elastic * 1.03


def test_fiber_catalog():
    from vfife.catalog import library
    lib = library()
    assert "beam2_fiber" in lib["elements"]
    assert set(lib["elements"]["beam2_fiber"]["compatible_materials"]) == {
        "rebar", "steel02", "piecewise_uniaxial", "linear_elastic",
        "concrete01", "concrete02",
        # the cyclic law of v0.1: degradation, ULCF, and the fracture
        # criterion the fibre latch of v0.1 counts
        "steel_cyclic"}


def _axial_case(tmpdir, length, n_elem, P=50000.0, E=200e9):
    """Cantilever fiber beam, n_elem elements, axial point load at the tip."""
    msh = pathlib.Path(tmpdir) / "axial.msh"
    nodes = "\n".join(f"{i+1} {i*length/n_elem:.10g} 0 0" for i in range(n_elem + 1))
    elems = "\n".join(f"{e+1} 1 2 1 1 {e+1} {e+2}" for e in range(n_elem))
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n"
                   f"{n_elem+1}\n{nodes}\n$EndNodes\n$Elements\n{n_elem}\n{elems}\n"
                   "$EndElements\n", encoding="utf-8")
    yaml_p = pathlib.Path(tmpdir) / "axial.yaml"
    yaml_p.write_text(f"""materials:
  steel: {{type: rebar, E: {E}, yield_strength: 1.0e12, hardening: 1.0e9, rho: 7850.0}}
sections:
  # 2 x 2 on purpose: this checks P*L/(EA), and A is the same however the
  # section is diced.
  rc: {{type: fiber, material: steel, shape: rect, width: 0.05, height: 0.10, ny: 2, nz: 2}}
assign:
  - {{region: all, section: rc}}
mesh:
  type: gmsh
  path: {msh}
regions:
  fixed: {{box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}}
  tip: {{box: [{length-0.01}, {length+0.01}, -0.01, 0.01, -0.01, 0.01]}}
constraints:
  - region: fixed
    dofs: [true, true, true, true, true, true]
loads:
  - region: tip
    vector: [ {P}, 0, 0 ]
solver:
  backend: numpy
  mode: relaxation
  n_steps: 1200
  # 1e-3 on a 50 kN load is 50 N of force balance; the assertion below is
  # on 2% of an elongation. 1e-4 made every load stage run to its budget
  # -- a stiff axial DOF converges slowly under relaxation -- for three
  # digits nothing here reads.
  tolerance: 1.0e-3
  # an elastic pull does not need the 20 stages a nonlinear material defaults to
  load_steps: 3
output:
  monitors:
    tip_disp: {{region: tip, dof: x, kind: displacement}}
""", encoding="utf-8")
    return str(yaml_p)


@pytest.mark.parametrize("length,n_elem", [(1.0, 1), (2.0, 4), (1.6, 4), (3.0, 6)])
def test_fiber_beam_axial_elongation_is_independent_of_element_length(length, n_elem):
    """P*L/(EA), whatever the element length -- the case that stayed hidden.

    `test_fiber_beam_axial_elastic` above already asserts exactly this, and it
    passed throughout a bug that made the axial internal force L0 times too
    large: the assembly used dEps/du = [-1, 1] where the strain (L - L0)/L0
    gives [-1/L0, 1/L0]. That test uses ONE element of length 1.0 m, and at
    L0 = 1.0 the missing factor is exactly 1. A correct assertion, blind for
    the only reason a correct assertion can be blind -- the geometry it was
    written on could not express the error.

    On real work the factor was not 1. case_11 (Tanaka & Park column, 400 mm
    elements) put 2048 kN through a section carrying an applied 819 kN, an
    axial load ratio of 0.5 f'c Ag where the case asks for 0.2, and the column
    came out 20% too strong against the measured hysteresis.

    So: several lengths, several element counts, and never a unit length.
    """
    import tempfile
    from vfife.api import prepare_case, run_case
    with tempfile.TemporaryDirectory() as td:
        prep = prepare_case(_axial_case(td, length, n_elem))
        assert prep.preflight["status"] == "PASS"
        out = run_case(prep)
        assert out.converged, f"L={length} n={n_elem} did not converge"
        area = 0.05 * 0.10
        expected = 50000.0 * length / (200e9 * area)
        tip = float(out.monitors["tip_disp"][-1])
        assert abs(tip - expected) / expected < 0.02, (
            f"L={length} n={n_elem} (L0={length/n_elem}): tip {tip:.4e} against "
            f"P*L/EA {expected:.4e}; ratio {tip/expected:.3f} -- a ratio near "
            f"{n_elem/length:.3g} means the 1/L0 in dEps/du is missing again")


@pytest.mark.parametrize("length,n_elem", [(2.0, 4), (1.6, 4)])
def test_fiber_beam_reports_section_force_not_its_integral(length, n_elem):
    """`axial_force` is a force in newtons, as it is for truss2 and beam2.

    test_truss2 asserts axial_force == 1000.0 for a 1000 N load and test_beam2
    asserts bending_z == 1000.0 for a 1000 N.m moment, so the convention across
    element families is the section action itself. The fiber path was
    accumulating `weight * L0 * [N, My, Mz]` and reporting that -- the integral
    along the element, i.e. L0 times the section force. The legacy secant path
    a few lines below in the same file always reported abs(N) directly, so the
    two halves of one file disagreed.

    It went unnoticed because the L0 error here cancelled the separate L0 error
    in the internal-force assembly: with both present, a 400 mm element under
    an applied 819 kN stored exactly 819 kN, the right answer for two wrong
    reasons. Fixing only one of them would have made the output look broken.
    """
    import tempfile
    from vfife.api import prepare_case, run_case
    with tempfile.TemporaryDirectory() as td:
        yp = _axial_case(td, length, n_elem)
        out_h5 = pathlib.Path(td) / "out.h5"
        prep = prepare_case(yp)
        prep.case.output.fields = ["displacement", "axial_force"]
        out = run_case(prep, output=str(out_h5))
        assert out.converged
        import h5py
        with h5py.File(out_h5) as f:
            last = sorted(f["steps"], key=int)[-1]
            axial = np.asarray(f[f"steps/{last}/axial_force"])
        assert axial.shape == (n_elem,)
        for e, value in enumerate(axial):
            assert value == pytest.approx(50000.0, rel=0.02), (
                f"element {e} of length {length/n_elem} reports {value:.1f} N for "
                f"an applied 50 kN; a value near {50000.0*length/n_elem:.0f} means "
                f"the reported action is still L0 times the section force")
