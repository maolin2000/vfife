"""The energy audit with damping in it.

Two defects, both in the bookkeeping and neither in the solution:

* THE DAMPING FORCE'S WORK WAS NEVER BOOKED. Rayleigh `alpha M v (+ beta K v)`
  and the viscous `zeta M v` act in every damped dynamic run, and the ledger
  counted neither, so their work came out as `energy_imbalance`: 0.54 and
  0.77 relative error on the Hangzhou Bay seismic runs, 93-100% of it this.
* THE FRAME, FIBRE AND TRUSS SOLVERS FED THE LEDGER ONE STEP LATE -- the
  forces of step n against the displacement increment of step n+1, the lag
  v0.1 had already removed from the continuum solver. On a 3-D seismic run
  it drifted to 15% of the external work by 6.5 s.

Each case below is a cantilever (a bar for the truss) under a suddenly
applied load in dynamic mode, so the load does work, the structure rings and
the damping takes the ringing out: external work = strain + kinetic +
dissipated has to hold throughout, and the dissipated part has to be most
of the external work's excess over the strain energy by the end.
"""
import pathlib

import h5py
import numpy as np
import pytest

from vfife.api import run_case

N, L = 8, 1.0


def _mesh(td):
    msh = pathlib.Path(td) / "m.msh"
    nodes = "\n".join(f"{i+1} {i*L/N} 0 0" for i in range(N + 1))
    els = "\n".join(f"{e+1} 1 2 1 1 {e+1} {e+2}" for e in range(N))
    msh.write_text(f"""$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
1 1 "all"
0 2 "fixed"
0 3 "loaded"
$EndPhysicalNames
$Nodes
{N+1}
{nodes}
$EndNodes
$Elements
{N+2}
{N+1} 15 2 2 2 1
{N+2} 15 2 3 3 {N+1}
{els}
$EndElements
""", encoding="utf-8")
    return msh


SECTIONS = {
    "beam2": ("steel: {type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850.0}",
              "s: {type: beam, material: steel, shape: general, area: 0.01, iy: 1.0e-5, iz: 1.0e-5, j: 2.0e-5}",
              "{region: all, section: s, orientation: [0, 0, 1]}",
              "- {region: fixed, dofs: [1, 1, 1, 1, 1, 1]}",
              "[0, 0, -2000.0]"),
    "fiber": ("steel: {type: steel02, E: 2.0e+11, Fy: 1.0e+10, b: 0.01, R0: 20.0, rho: 7850.0}",
              "s: {type: fiber, material: steel, shape: rect, width: 0.1, height: 0.1, ny: 4, nz: 4}",
              "{region: all, section: s, orientation: [0, 0, 1]}",
              "- {region: fixed, dofs: [1, 1, 1, 1, 1, 1]}",
              "[0, 0, -2000.0]"),
    "truss": ("steel: {type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850.0}",
              "s: {type: bar, material: steel, shape: general, area: 1.0e-4}",
              "{region: all, section: s}",
              "- {region: fixed, dofs: [1, 1, 1]}\n- {region: all, dofs: [0, 1, 1]}",
              "[2000.0, 0, 0]"),
}


def _run(td, kind, damping, total):
    td = pathlib.Path(td); td.mkdir(parents=True, exist_ok=True)
    mat, sec, assign, cons, load = SECTIONS[kind]
    case = td / "c.yaml"
    case.write_text(f"""
name: damped_{kind}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {_mesh(td)}}}
materials:
  {mat}
sections:
  {sec}
assign:
- {assign}
constraints:
{cons}
loads:
- {{region: loaded, vector: {load}}}
steps: [{{name: s}}]
solver:
  mode: dynamic
  backend: numpy
  total_time: {total}
  damping: {damping}
output:
  path: {td / 'o.h5'}
  every: 1000000
  monitors: {{}}
""", encoding="utf-8")
    r = run_case(str(case), progress_every=50)
    assert r.status == "COMPLETED_DYNAMIC", r.message
    with h5py.File(r.result_path) as h:
        d = {k: h["diagnostics"][k][...] for k in h["diagnostics"]}
    return d


# (kind, damping, total time): long enough for the damping to take out most
# of the ringing -- the first bending period is ~0.03 s, the axial one ~0.8 ms
CASES = [
    ("beam2", "{type: rayleigh, alpha: 60.0}", 0.15),
    ("beam2", "{type: viscous, coeff: 60.0}", 0.15),
    ("fiber", "{type: rayleigh, alpha: 60.0}", 0.15),
    ("truss", "{type: rayleigh, alpha: 3000.0}", 0.004),
    ("truss", "{type: viscous, coeff: 3000.0}", 0.004),
]


@pytest.mark.parametrize("kind,damping,total", CASES)
def test_the_audit_closes_with_the_damping_booked(tmp_path, kind, damping, total):
    d = _run(tmp_path, kind, damping, total)
    W, U, K, D = (d[k][-1] for k in ("external_work", "internal_work",
                                      "kinetic_energy", "dissipated_energy"))
    err = d["relative_energy_error"]
    assert W > 0.0
    # the damping took most of the ringing: under a suddenly applied load the
    # external work at rest is 2 x the strain energy, the other half is what
    # the damping must have dissipated
    assert D > 0.5 * (W - U), (W, U, K, D)
    # past the start-up (the first snapshots compare work to u_n against a
    # kinetic energy at v_(n+1/2), on totals that are still tiny) the
    # balance holds to 1e-5 on the frames and 5e-4 on the truss, which gets
    # six snapshots in all; at the end it is 1e-9 and 3e-5
    n = len(err)
    assert np.abs(err[max(1, n // 10):]).max() < 1e-3, np.abs(err[n // 10:]).max()
    assert abs(err[-1]) < 1e-4, err[-1]
