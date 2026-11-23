"""The axial members of a mixed line model, evaluated all at once.

`MixedStructuralSolver.internal_force` looped over its trusses one element
at a time, on the reasoning that "trusses and connectors are few". On a cable
bridge they are not: the Xihoumen model has 346 cable members, and the loop
was 35 of a 39 ms force evaluation -- the finite-difference modal tangent,
2 n of those evaluations, went from about a minute to over six.

The batched path is pinned here to `_internal_force_reference`, the
element-at-a-time version, on the model that exercises everything the loop
had to get right: a prestressed cable (unstressed length below the mesh
length), a tension-only member pushed into compression (it must carry
nothing), frame members with finite rotations around them, and a group
with its own modulus, so `member_E` is read per element.
"""
import pathlib

import numpy as np

from vfife.api import prepare_case
from vfife.core.rotations import exp_map
from vfife.core.structural import make_structural_solver


def _case(td):
    td = pathlib.Path(td)
    # a portal: two columns and a beam, a prestressed diagonal cable, and a
    # tension-only tie that the displacement below will put into compression
    pts = [(0, 0, 0), (0, 0, 3), (4, 0, 3), (4, 0, 0), (2, 0, 3)]
    lines = [("frame", 0, 1), ("frame", 1, 4), ("frame", 4, 2), ("frame", 2, 3),
             ("cable", 0, 2), ("tie", 1, 3), ("strut", 0, 4)]
    tags = {"frame": 1, "cable": 2, "tie": 3, "strut": 4}
    msh = td / "m.msh"
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "5",
           '1 1 "frame"', '1 2 "cable"', '1 3 "tie"', '1 4 "strut"', '0 9 "base"',
           "$EndPhysicalNames", "$Nodes", str(len(pts))]
    out += [f"{i+1} {x} {y} {z}" for i, (x, y, z) in enumerate(pts)]
    out += ["$EndNodes", "$Elements", str(len(lines) + 2)]
    out += ["1 15 2 9 9 1", "2 15 2 9 9 4"]
    out += [f"{k+3} 1 2 {tags[g]} {tags[g]} {a+1} {b+1}" for k, (g, a, b) in enumerate(lines)]
    out += ["$EndElements", ""]
    msh.write_text("\n".join(out), encoding="utf-8")
    p = td / "c.yaml"
    p.write_text(f"""
name: truss_batch
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  steel: {{type: linear_elastic, E: 2.0e+11, nu: 0.3, rho: 7850.0}}
  soft:  {{type: linear_elastic, E: 7.0e+10, nu: 0.3, rho: 2700.0}}
sections:
  col:   {{type: beam, material: steel, shape: general, area: 0.01, iy: 1.0e-4, iz: 2.0e-4, j: 1.5e-4}}
  cable: {{type: bar, material: steel, shape: general, area: 5.0e-4}}
  tie:   {{type: bar, material: steel, shape: general, area: 3.0e-4}}
  strut: {{type: bar, material: soft, shape: general, area: 8.0e-4}}
assign:
  - {{region: frame, section: col, orientation: [0, 1, 0]}}
  - {{region: cable, section: cable, prestress: 3.0e+8}}
  - {{region: tie,   section: tie, tension_only: true}}
  - {{region: strut, section: strut}}
constraints: [{{region: base, dofs: [1, 1, 1, 1, 1, 1]}}]
loads: []
steps: [{{name: s, load_steps: 1}}]
solver: {{backend: numpy, mode: relaxation, tolerance: 1.0e-6, n_steps: 10}}
output: {{path: {td/'o.h5'}, monitors: {{}}}}
""", encoding="utf-8")
    return str(p)


def _solver(path):
    prep = prepare_case(path)
    return make_structural_solver(prep.case.model_dump(mode="python"), prep.mesh, prep.material)


def test_the_batched_axial_forces_are_the_element_loop(tmp_path):
    sol = _solver(_case(tmp_path))
    rng = np.random.default_rng(7)
    sol.q = rng.normal(size=sol.q.shape) * 2e-3
    sol.orient = exp_map(sol.dofs.view(sol.q)[:, 3:])
    f_b, a_b = sol.internal_force()
    f_r, a_r = sol._internal_force_reference()
    assert np.max(np.abs(f_b - f_r)) < 1e-12 * np.max(np.abs(f_r))
    assert np.max(np.abs(a_b - a_r)) < 1e-12 * np.max(np.abs(a_r))


def test_the_batch_honours_prestress_slack_and_a_groups_own_modulus(tmp_path):
    """Not only equal to the loop -- equal to the closed form, member by member."""
    sol = _solver(_case(tmp_path))
    sol.internal_force()
    kinds = sol.member_kind
    trusses = np.flatnonzero(kinds == "truss2")
    N = sol.end_actions[trusses, 6]
    # at u = 0 the prestressed cable carries its prestress, the others nothing
    cable = trusses[0]
    assert np.isclose(N[0], 3.0e8 * 5.0e-4, rtol=1e-12)
    assert np.all(N[1:] == 0.0)
    # squeeze the tie: node 2 (index 1) toward node 4 (index 3) -- it goes slack
    sol.q[:] = 0.0
    sol.dofs.view(sol.q)[1, 0] = 0.01
    sol.internal_force()
    tie = trusses[1]
    assert sol.end_actions[tie, 6] == 0.0
    # stretch the strut: its force uses the SOFT group's modulus
    sol.q[:] = 0.0
    L0 = sol.length[trusses[2]]
    sol.dofs.view(sol.q)[4, 0] = 1e-3               # node 5 moves along +x
    sol.internal_force()
    x = sol.mesh.nodes[[0, 4]].copy(); x[1, 0] += 1e-3
    L = np.linalg.norm(x[1] - x[0])
    assert np.isclose(sol.end_actions[trusses[2], 6], 7.0e10 * 8.0e-4 * (L - L0) / L0, rtol=1e-12)
    assert cable != tie
