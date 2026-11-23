"""Mode shapes as ParaView time series, and the condensed dofs in them.

Two things are pinned here. The writer: each frame is the reference mesh
displaced by scale * phi * sin(2 pi t), so the quarter-period frame IS the
mode shape at the stated scale and the first frame is the undeformed mesh;
the all-modes collection puts mode k on [k-1, k). And `modal.analyse`,
which left a condensed (massless but stiff) dof at zero in every shape: the
eigenvalues never showed it, an animation shows it at once -- the node
snaps back to where it started while its neighbours swing.
"""
import pathlib
import re

import meshio
import numpy as np
import pytest

from vfife import modal
from vfife.api import prepare_case
from vfife.core.structural import make_structural_solver
from vfife.post import modes_vtk

N, L = 6, 1.0


def _case(td, tip_rho=7850.0):
    """A cantilever of six beam2 elements along x; the last element is its own
    group with its own density, so it can be made massless."""
    td = pathlib.Path(td); td.mkdir(parents=True, exist_ok=True)
    msh = td / "m.msh"
    nodes = "\n".join(f"{i+1} {i*L} 0 0" for i in range(N + 1))
    els = "\n".join(f"{e+1} 1 2 {1 if e < N-1 else 2} {1 if e < N-1 else 2} {e+1} {e+2}"
                    for e in range(N))
    msh.write_text(f"""$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
1 1 "body"
1 2 "tip"
0 3 "root"
$EndPhysicalNames
$Nodes
{N+1}
{nodes}
$EndNodes
$Elements
{N+1}
{N+1} 15 2 3 3 1
{els}
$EndElements
""", encoding="utf-8")
    p = td / "c.yaml"
    p.write_text(f"""
name: cantilever
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  steel: {{type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850.0}}
  tipm:  {{type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: {tip_rho}}}
sections:
  s: {{type: beam, material: steel, shape: general, area: 0.01, iy: 1.0e-5, iz: 2.0e-5, j: 1.0e-5}}
  t: {{type: beam, material: tipm,  shape: general, area: 0.01, iy: 1.0e-5, iz: 2.0e-5, j: 1.0e-5}}
assign:
  - {{region: body, section: s, orientation: [0, 0, 1]}}
  - {{region: tip,  section: t, orientation: [0, 0, 1]}}
constraints: [{{region: root, dofs: [1, 1, 1, 1, 1, 1]}}]
loads: []
steps: [{{name: s, load_steps: 1}}]
solver: {{backend: numpy, mode: relaxation, tolerance: 1.0e-6, n_steps: 10}}
output: {{path: {td/'o.h5'}, monitors: {{}}}}
""", encoding="utf-8")
    return str(p)


def _modes(path, n=4):
    prep = prepare_case(path)
    sol = make_structural_solver(prep.case.model_dump(mode="python"), prep.mesh, prep.material)
    return sol, modal.analyse(sol, n_modes=n)


def test_the_quarter_period_frame_is_the_mode_shape_at_the_stated_scale(tmp_path):
    sol, res = _modes(_case(tmp_path / "c"))
    out = tmp_path / "pv"
    modes_vtk.write(sol, res, out, frames=8, scale=0.5, labels=["a", "b", "c", "d"],
                    log=lambda *_: None)
    x0 = sol.mesh.nodes
    U = res.shapes.reshape(len(res.frequencies), sol.nn, 6)[:, :, :3]
    for k in range(1, 5):
        tag = f"mode{k:02d}_{'abcd'[k-1]}"
        f0 = meshio.read(out / "modes" / f"{tag}_000.vtu")
        f2 = meshio.read(out / "modes" / f"{tag}_002.vtu")        # phase 1/4
        assert np.allclose(f0.points, x0, atol=1e-12)
        assert np.allclose(f2.points - x0, 0.5 * U[k - 1], atol=1e-12)
        assert np.isclose(np.abs(f2.point_data["displacement"]).max(), 0.5)
        pvd = (out / f"modes_{tag}.pvd").read_text(encoding="utf-8")
        assert len(re.findall("<DataSet", pvd)) == 8


def test_the_all_modes_collection_puts_mode_k_on_k_minus_1_to_k(tmp_path):
    sol, res = _modes(_case(tmp_path / "c"), n=3)
    out = tmp_path / "pv"
    modes_vtk.write(sol, res, out, frames=4, log=lambda *_: None)
    steps = [float(t) for t in re.findall(r'timestep="([^"]+)"', (out / "modes_all.pvd").read_text(encoding="utf-8"))]
    assert steps == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
    table = (out / "modes.txt").read_text(encoding="utf-8").splitlines()
    assert len([r for r in table if re.match(r"\s+\d+\s", r)]) == 3
    assert any(r.startswith("group:") for r in table)


def test_the_cells_carry_the_meshs_own_groups(tmp_path):
    sol, res = _modes(_case(tmp_path / "c"), n=1)
    out = tmp_path / "pv"
    modes_vtk.write(sol, res, out, frames=2, log=lambda *_: None)
    m = meshio.read(next((out / "modes").glob("mode01_000.vtu")))
    g = m.cell_data["group"][0]
    legend = dict((int(i), n.strip()) for i, n in (kv.split("=") for kv in
                  (out / "modes.txt").read_text(encoding="utf-8").split("group:")[1].splitlines()[0].split(",")))
    assert [legend[i] for i in g] == ["body"] * (N - 1) + ["tip"]


def test_a_condensed_dof_follows_its_neighbours_instead_of_staying_at_zero(tmp_path):
    """The last element is massless (rho 1e-12: a material refuses 0, it
    computes a wave speed), so the tip node carries no mass: it is
    condensed out of the eigenproblem. A massless end segment carries no
    inertia load, so it has no shear and no moment and stays STRAIGHT:
    u_tip = u_(N-1) + theta_(N-1) L. Before this stage the tip was zero."""
    sol, res = _modes(_case(tmp_path / "c", tip_rho=1.0e-12), n=3)
    assert res.n_condensed > 0
    V = res.shapes.reshape(len(res.frequencies), sol.nn, 6)
    for k in range(3):
        u = V[k]
        # first bending mode about z deflects in y (theta_z), about y in z (theta_y)
        for comp_u, comp_t, sign in ((1, 5, +1), (2, 4, -1)):
            pred = u[N - 1, comp_u] + sign * u[N - 1, comp_t] * L
            scale = np.abs(u[:, comp_u]).max()
            if scale < 1e-6:
                continue
            assert u[N, comp_u] == pytest.approx(pred, abs=1e-8 * max(scale, 1.0))
            assert abs(u[N, comp_u]) > 0.1 * scale        # and it is not zero


def test_the_cli_writes_them_on_request(tmp_path, capsys):
    from vfife.cli import main
    case = _case(tmp_path / "c")
    rc = main(["modal", case, "--modes", "2", "-o", str(tmp_path / "m.h5"),
               "--paraview", str(tmp_path / "pv"), "--frames", "4"])
    assert rc == 0
    assert (tmp_path / "pv" / "modes_all.pvd").exists()
    assert len(list((tmp_path / "pv" / "modes").glob("*.vtu"))) == 8
