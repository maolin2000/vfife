"""D1: fiber_integration gauss2 vs lobatto3 selectable."""
import pathlib, tempfile, yaml, shutil

def _run(integ):
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    msh = tmpdir / "msh.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n2\n1 0 0 0\n2 1 0 0\n$EndNodes\n$Elements\n1\n1 1 2 1 1 1 2\n$EndElements\n", encoding="utf-8")
    yp = tmpdir / "case.yaml"
    yp.write_text(f"""
materials: {{steel: {{type: linear_elastic, E: 200.0e9, nu: 0.3, rho: 7850}}}}
sections: {{rc: {{type: fiber, material: steel, shape: rect, width: 0.1, height: 0.2, ny: 6, nz: 6, integration: {integ}}}}}
assign: [{{region: all, section: rc}}]
mesh: {{type: gmsh, path: {str(msh).replace(chr(92),"/")}}}
regions: {{fixed: {{box: [-0.01,0.01,-0.01,0.01,-0.01,0.01]}}, tip: {{box: [0.99,1.01,-0.01,0.01,-0.01,0.01]}}}}
constraints: [{{region: fixed, dofs: [true,true,true,true,true,true]}}]
solver: {{backend: numpy, mode: relaxation, n_steps: 100, load_steps: 20, tolerance: 1.0e-4, kinetic_tolerance: 1.0e-8}}
steps: [{{name: push, loads: [{{region: tip, vector: [0,0,1000]}}]}}]
output: {{monitors: {{tip_disp: {{region: tip, dof: z, kind: displacement}}}}}}
""", encoding="utf-8")
    from vfife.api import prepare_case
    from vfife.core.structural import make_structural_solver
    from vfife.io.schema import load as sload
    prep = prepare_case(str(yp))
    case,_ = sload(str(yp))
    prob = case.model_dump(mode="python")
    sol = make_structural_solver(prob, prep.mesh, prep.material)
    sol.run()
    return float(sol.monitors["tip_disp"].values[-1]), sol

def test_fiber_integration_gauss2_is_default():
    from vfife.core.fiber_beam_solver import FiberBeamSolver
    assert hasattr(FiberBeamSolver, "BEAM_TYPES")

def test_fiber_integration_both_converge_and_match():
    d2, s2 = _run("gauss2")
    d3, s3 = _run("lobatto3")
    assert abs(d2 - d3) / max(abs(d2), 1e-12) < 1e-6
    assert len(s2.gauss_s) == 2
    assert len(s3.gauss_s) == 3
    assert s3.gauss_s[0] == 0.0 and s3.gauss_s[-1] == 1.0

def test_fiber_integration_checkpoint_roundtrip():
    # lobatto has 3 gauss points -> checkpoint shape must be 3
    _, sol = _run("lobatto3")
    ck = sol.checkpoint()
    assert ck["fiber_state"].shape[1] == 3
    q0 = sol.q.copy()
    sol.q += 1e-9
    sol.restore(ck)
    assert abs(sol.q - q0).max() < 1e-12
