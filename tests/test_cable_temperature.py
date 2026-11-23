"""A cable expands too.

`BeamSolver._thermal_deformation` (now `_thermal_load`) kept the beam members of a temperature
region and dropped the rest without a word, so on a `mixed_line` model a
temperature on a region holding girder and stays heated the girder and ran,
converged and reported success with the stays at their erection temperature
(scan item 8). An axial member now takes its free thermal
elongation alpha dT Lu off the stretch before E A sees it.

The model: two beam elements (a stiff cantilever stub) and two bar elements
in line along x. The closed forms are the two that define a free strain:
held at both ends a bar carries -E A alpha dT and does not move; free at one
end it grows alpha dT L and carries nothing.
"""
import pathlib

import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.structural import make_structural_solver

E_S, E_C, A_C, ALPHA, DT = 2.1e11, 1.95e11, 1.0e-3, 1.2e-5, 50.0


def _case(td, bar_end_fixed, temperature="{uniform: 50.0}", tension_only=False, lu=None):
    td = pathlib.Path(td); td.mkdir(parents=True, exist_ok=True)
    msh = td / "m.msh"
    msh.write_text("""$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
1 1 "girder"
1 2 "stay"
$EndPhysicalNames
$Nodes
5
1 0 0 0
2 0.5 0 0
3 1.0 0 0
4 1.5 0 0
5 2.0 0 0
$EndNodes
$Elements
4
1 1 2 1 1 1 2
2 1 2 1 1 2 3
3 1 2 2 2 3 4
4 1 2 2 2 4 5
$EndElements
""", encoding="utf-8")
    end = "[1, 1, 1, 0, 0, 0]" if bar_end_fixed else "[0, 1, 1, 0, 0, 0]"
    stay = "{region: stay, section: s" + (", tension_only: true" if tension_only else "") + \
           (f", unstressed_lengths: [{lu}, {lu}]" if lu is not None else "") + "}"
    p = td / "c.yaml"
    p.write_text(f"""
name: cable_temperature
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  steel: {{type: linear_elastic, E: {E_S}, nu: 0.3, rho: 7850.0, thermal_expansion: {ALPHA}}}
  strand: {{type: linear_elastic, E: {E_C}, nu: 0.3, rho: 7850.0, thermal_expansion: {ALPHA}}}
sections:
  g: {{type: beam, material: steel, shape: general, area: 0.05, iy: 1.0e-3, iz: 1.0e-3, j: 2.0e-3}}
  s: {{type: bar, material: strand, shape: general, area: {A_C}}}
assign:
  - {{region: girder, section: g, orientation: [0, 0, 1]}}
  - {stay}
regions:
  root: {{box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}}
  mid:  {{box: [0.99, 1.01, -0.01, 0.01, -0.01, 0.01]}}
  far:  {{box: [1.99, 2.01, -0.01, 0.01, -0.01, 0.01]}}
  bar_inner: {{box: [1.49, 1.51, -0.01, 0.01, -0.01, 0.01]}}
constraints:
  - {{region: root, dofs: [1, 1, 1, 1, 1, 1]}}
  - {{region: mid, dofs: [1, 1, 1, 1, 1, 1]}}
  - {{region: bar_inner, dofs: [0, 1, 1, 1, 1, 1]}}
  - {{region: far, dofs: {end}}}
loads:
  - {{region: stay, temperature: {temperature}}}
steps: [{{name: heat, load_steps: 1}}]
solver: {{backend: numpy, mode: relaxation, tolerance: 1.0e-9, n_steps: 20000}}
output:
  path: {td / 'o.h5'}
  monitors: {{u: {{kind: displacement, region: far, dof: x}}}}
""", encoding="utf-8")
    return str(p)


def _solve(path):
    prep = prepare_case(path)
    sol = make_structural_solver(prep.case.model_dump(mode="python"), prep.mesh, prep.material)
    sol.run(output_every=10**9, progress_every=10**9)
    sol.internal_force()
    return sol, sol.end_actions


def test_a_bar_held_at_both_ends_carries_minus_EA_alpha_dT(tmp_path):
    sol, ends = _solve(_case(tmp_path, True))
    N = ends[2:, 6]
    assert N == pytest.approx(-E_C * A_C * ALPHA * DT, rel=1e-6)
    # and the girder, held at both of its ends and not heated, carries nothing
    assert np.abs(ends[:2]).max() < 1e-6 * E_C * A_C * ALPHA * DT


def test_a_bar_free_at_one_end_grows_alpha_dT_L_and_carries_nothing(tmp_path):
    sol, ends = _solve(_case(tmp_path, False))
    u = sol.dofs.view(sol.q)[4, 0]
    assert u == pytest.approx(ALPHA * DT * 1.0, rel=1e-6)
    assert np.abs(ends[2:, 6]).max() < 1e-6 * E_C * A_C * ALPHA * DT


def test_a_pretensioned_stay_loses_EA_alpha_dT_of_its_tension(tmp_path):
    """A stay 0.1% short of its chord carries E A 1e-3 cold; 50 K takes
    E A alpha dT = 0.6 of that off it, measured from its unstressed length."""
    lu = 0.5 / 1.001
    sol, ends = _solve(_case(tmp_path, True, lu=lu, tension_only=True))
    N = ends[2:, 6]
    assert N == pytest.approx(E_C * A_C * (0.5 - lu - ALPHA * DT * lu) / lu, rel=1e-6)


def test_a_profile_on_a_bar_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="bar section has no shape"):
        _solve(_case(tmp_path, True,
                     temperature="{profile: [[-0.1, 0.0], [0.1, 10.0]], axis: z}"))
