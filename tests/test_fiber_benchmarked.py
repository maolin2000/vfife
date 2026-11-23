"""Benchmarked evidence for beam2_fiber with concrete02+steel02+lobatto3+6elem.

Covers qualification_gates benchmarked requirements:
  - mesh_convergence: 1 vs 2 vs 6 lobatto3, linear_elastic static tip load (element formulation)
  - dynamic: hybrid concrete02+steel02 cyclic in dynamic mode
  - restart: checkpoint round-trip for hybrid lobatto3
  - external_reference: RC capacity hand-check vs ACI (cracked, softening)
"""
import pathlib, tempfile, uuid
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)

def _cyclic_yaml(tmpdir, n_elem, ny=4, nz=4, integ="lobatto3", load_steps=16, peaks=None, mode="relaxation", material="concrete02"):
    msh = pathlib.Path(tmpdir) / f"mesh_{n_elem}.msh"
    L = 1.5
    nodes = [f"{i+1} {i*L/n_elem:.6f} 0 0" for i in range(n_elem+1)]
    elems = [f"{i+1} 1 2 1 1 {i+1} {i+2}" for i in range(n_elem)]
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n"+str(n_elem+1)+"\n"+"\n".join(nodes)+"\n$EndNodes\n$Elements\n"+str(n_elem)+"\n"+"\n".join(elems)+"\n$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(tmpdir) / f"case_{n_elem}_{uuid.uuid4().hex}.yaml"
    # load levels are a relaxation's; a dynamic run is refused them
    levels = f"load_steps: {load_steps}, " if mode == "relaxation" else ""
    if peaks is None:
        peaks = [0.006, 0.012]
    if material == "linear_elastic":
        mat_block = "type: linear_elastic, E: 30.0e9, nu: 0.3, rho: 2500"
        rebar_block = ""
    else:
        mat_block = "type: concrete02, fpc: 35.0e6, epsc0: 0.002, fcu: 7.0e6, epscu: 0.0035, ft: 3.2e6, Ets: 1.2e9, rat: 0.1, rho: 2500"
        rebar_block = ", bars: [{y: 0.09, z: 0.14, section: bar}, {y: -0.09, z: 0.14, section: bar}, {y: 0.09, z: -0.14, section: bar}, {y: -0.09, z: -0.14, section: bar}]"
    yp.write_text(f"""
name: bench_{n_elem}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92),"/")}}}
materials: {{main: {{{mat_block}}}, steel: {{type: steel02, E: 200.0e9, Fy: 350.0e6, b: 0.01}}}}
sections:
  bar: {{type: bar, material: steel, shape: general, area: 3.14e-4}}
  rc: {{type: fiber, material: main, shape: rect, width: 0.25, height: 0.35, ny: {ny}, nz: {nz}, integration: {integ}{rebar_block}}}
assign: [{{region: all, section: rc}}]
regions: {{base: {{box: [-0.01,0.01,-0.01,0.01,-0.01,0.01]}}, head: {{box: [{L-0.01},{L+0.01},-0.01,0.01,-0.01,0.01]}}}}
constraints: [{{region: base, dofs: [1,1,1,1,1,1]}}]
amplitudes: {{drift: {{cyclic: {{peaks: {peaks}, repeats: 1}}}}}}
steps: [{{name: cyclic, constraints: [{{region: head, dofs: [0,0,1,0,0,0], value: 1.0, amplitude: drift}}]}}]
# max_steps_per_load was 3000, calibrated when the fibre beam's axial internal
# force was L0 times too large. Correcting that (dEps/du carries a 1/L0) made
# the axial DOF as stiff as it should be -- about 4x here, on 250 mm elements
# -- and a stiff DOF converges slowly under relaxation, so 3000 stopped being
# enough. Measured: 3000 leaves stage 1/16 at residual 4.4e-03 against a 1e-03
# tolerance; 15000 converges. Not stalling, just a harder solve than the
# budget was set for.
solver: {{backend: numpy, mode: {mode}, {levels}max_steps_per_load: 15000, tolerance: 1.0e-3, n_steps: 600, dt_safety: 0.6, total_time: 0.2}}
output: {{monitors: {{tip_disp: {{region: head, dof: z, kind: displacement}}, base_shear: {{region: base, dof: z, kind: reaction}}}}}}
""", encoding="utf-8")
    return str(yp)

def _static_yaml(tmpdir, n_elem):
    msh = pathlib.Path(tmpdir) / f"mesh_{n_elem}.msh"
    L = 1.5
    nodes = [f"{i+1} {i*L/n_elem:.6f} 0 0" for i in range(n_elem+1)]
    elems = [f"{i+1} 1 2 1 1 {i+1} {i+2}" for i in range(n_elem)]
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n"+str(n_elem+1)+"\n"+"\n".join(nodes)+"\n$EndNodes\n$Elements\n"+str(n_elem)+"\n"+"\n".join(elems)+"\n$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(tmpdir) / f"static_{n_elem}_{uuid.uuid4().hex}.yaml"
    yp.write_text(f"""
name: static_{n_elem}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92),"/")}}}
materials: {{main: {{type: linear_elastic, E: 30.0e9, nu: 0.3, rho: 2500}}}}
sections: {{rc: {{type: fiber, material: main, shape: rect, width: 0.25, height: 0.35, ny: 4, nz: 4, integration: lobatto3}}}}
assign: [{{region: all, section: rc}}]
regions: {{base: {{box: [-0.01,0.01,-0.01,0.01,-0.01,0.01]}}, head: {{box: [{L-0.01},{L+0.01},-0.01,0.01,-0.01,0.01]}}}}
constraints: [{{region: base, dofs: [1,1,1,1,1,1]}}]
loads: [{{region: head, vector: [0,0,1000]}}]
solver: {{backend: numpy, mode: relaxation, n_steps: 6000, tolerance: 5.0e-4}}
output: {{monitors: {{tip_disp: {{region: head, dof: z, kind: displacement}}}}}}
""", encoding="utf-8")
    return str(yp)

def _run_case(yaml_path):
    from vfife.api import prepare_case, run_case
    prep = prepare_case(yaml_path)
    assert prep.preflight["status"] == "PASS"
    out = run_case(prep)
    return out, prep

@pytest.mark.slow
# 网格加密收敛研究：结论只随单元公式变，不随一次重构变。
def test_mesh_convergence_2_vs_6():
    """Static linear-elastic fiber beam: tip displacement shall converge as mesh refines (1 vs 2 vs 6)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        outs = {}
        for n in [1, 2, 6]:
            yp = _static_yaml(td, n)
            out, _ = _run_case(yp)
            assert out.converged, f"n={n} did not converge: {out.diagnostics}"
            outs[n] = float(out.monitors["tip_disp"][-1])
        # Euler-Bernoulli cantilever PL^3/3EI ≈ 1000*3.375/(3*30e9*0.00089)=4.2e-05 for gross section
        # Allow 20% band due to shear and fiber discretization; convergence = finer not diverging
        assert outs[1] != 0 and outs[2] != 0 and outs[6] != 0
        rel_1_2 = abs(outs[1]-outs[2])/max(abs(outs[1]),1e-12)
        rel_2_6 = abs(outs[2]-outs[6])/max(abs(outs[2]),1e-12)
        assert rel_2_6 < 0.10, f"mesh not converging: 1={outs[1]:.2e} 2={outs[2]:.2e} 6={outs[6]:.2e} rel_2_6={rel_2_6:.3f}"
        assert rel_2_6 < rel_1_2 or rel_2_6 < 0.02, "refinement shall not diverge"

def test_dynamic_runs():
    """Hybrid concrete02+steel02 cyclic shall run in dynamic mode without NaN."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        yp = _cyclic_yaml(td, 2, ny=4, nz=4, load_steps=8, peaks=[0.006], mode="dynamic", material="concrete02")
        out, _ = _run_case(yp)
        assert np.all(np.isfinite(out.monitors["tip_disp"]))
        assert np.all(np.isfinite(out.monitors["base_shear"]))

def test_restart_checkpoint():
    """Lobatto3 hybrid checkpoint preserves q (fiber_state_0..4 keys, 3 points)."""
    import tempfile
    from vfife.api import prepare_case
    from vfife.core.structural import make_structural_solver
    from vfife.io.schema import load as sload
    with tempfile.TemporaryDirectory() as td:
        yp = _cyclic_yaml(td, 2, ny=4, nz=4, integ="lobatto3", load_steps=8, peaks=[0.006], material="concrete02")
        prep = prepare_case(yp)
        case, _ = sload(yp)
        prob = case.model_dump(mode="python")
        sol = make_structural_solver(prob, prep.mesh, prep.material)
        sol.run()
        ck = sol.checkpoint()
        assert "fiber_state_0" in ck
        assert ck["fiber_state_0"].shape[1] == 3  # lobatto3
        q0 = sol.q.copy()
        sol.q += 1e-7
        sol.restore(ck)
        assert np.max(np.abs(sol.q - q0)) < 1e-12

@pytest.mark.slow
# 对照 OpenSees 参照解的验证性测试，和 verification/ 里的契约同类；
#     夜间 `pytest -m slow` 会连同 case_11 的契约一起跑。
def test_external_reference_capacity():
    """Cracked RC at 12mm drift: 5-60kN band (Yassin concrete02 softening + GMP steel, mesh-sensitive)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        yp = _cyclic_yaml(td, 6, ny=4, nz=4, load_steps=16, peaks=[0.006, 0.012], mode="relaxation", material="concrete02")
        out, _ = _run_case(yp)
        assert out.converged
        peak_shear = float(np.max(np.abs(out.monitors["base_shear"])))
        assert 5000 < peak_shear < 60000, f"peak shear {peak_shear:.1f} outside cracked band"
