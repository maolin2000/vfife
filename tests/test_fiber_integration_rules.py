"""The fibre beam's integration rules along the member: gauss2, lobatto3,
lobatto4, lobatto5.

An elastic member's strain energy is quadratic in the Hermite curvature,
so every rule integrates it exactly and the four give the same answer to
round-off; what a Lobatto rule buys is a station AT the ends and, with
five, one a quarter-way in -- where a hinge sits. The kernel is compiled
per station count and must match the NumPy path at every one of them.
"""
import pathlib

import numpy as np
import pytest

from vfife.api import _with_element, prepare_case, run_case
from vfife.core.fiber_beam_solver import INTEGRATION_RULES
from vfife.core.structural import make_structural_solver

L, W, H = 4.0, 0.4, 0.2


def _case(td, rule, fields="[displacement, rotation, fiber_stress]", concrete=None, bars=""):
    msh = pathlib.Path(td) / "cant.msh"
    n = 4
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "3",
           '0 1 "base"', '0 2 "tip"', '1 3 "beam"', "$EndPhysicalNames", "$Nodes", str(n + 1)]
    out += [f"{k + 1} {L * k / n:.12g} 0 0" for k in range(n + 1)]
    out += ["$EndNodes", "$Elements", str(n + 2), "1 15 2 1 1 1", f"2 15 2 2 2 {n + 1}"]
    out += [f"{k + 3} 1 2 3 3 {k + 1} {k + 2}" for k in range(n)]
    out += ["$EndElements", ""]
    msh.write_text("\n".join(out), encoding="utf-8")
    concrete = concrete or "{type: linear_elastic, E: 2.0e11, nu: 0.0, rho: 7850}"
    yp = pathlib.Path(td) / f"cant_{rule}.yaml"
    yp.write_text(f"""
name: cant_{rule}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  s: {concrete}
  r: {{type: steel02, E: 2.0e11, Fy: 4.0e8, b: 0.01, rho: 7850}}
sections:
  bar: {{type: bar, material: r, shape: circle, d: 0.02}}
  m: {{type: fiber, material: s, shape: rect, width: {W}, height: {H}, ny: 8, nz: 4,
       integration: {rule}{bars}}}
assign: [{{region: all, section: m}}]
regions: {{}}
constraints:
  - {{region: base, dofs: [1, 1, 1, 1, 1, 1]}}
loads:
  - {{region: tip, vector: [0, 0, -1.0e5]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, max_steps_per_load: 20000,
          stage_ramp_iterations: 1, tolerance: 1.0e-7, kinetic_tolerance: 1.0e-10, n_steps: 50}}
output:
  path: {pathlib.Path(td) / f'cant_{rule}.h5'}
  fields: {fields}
  monitors:
    w_tip: {{kind: displacement, region: tip, dof: z}}
""", encoding="utf-8")
    return str(yp)


def _solver(path):
    prep = prepare_case(path)
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


def test_the_rules_are_exact_and_sum_to_one():
    for name, (S, Wt) in INTEGRATION_RULES.items():
        assert Wt.sum() == pytest.approx(1.0, abs=1e-15)
        deg = 3 if name == "gauss2" else 2 * len(S) - 3
        for k in range(deg + 1):
            assert np.sum(Wt * S ** k) == pytest.approx(1.0 / (k + 1), abs=1e-14)
    assert list(INTEGRATION_RULES["lobatto5"][0]) == pytest.approx(
        [0.0, 0.5 * (1 - np.sqrt(3 / 7)), 0.5, 0.5 * (1 + np.sqrt(3 / 7)), 1.0])
    assert INTEGRATION_RULES["lobatto5"][0][0] == 0.0 and INTEGRATION_RULES["lobatto5"][0][-1] == 1.0


@pytest.mark.parametrize("rule", ["lobatto3", "lobatto4", "lobatto5"])
def test_an_elastic_cantilever_is_the_same_under_every_rule(tmp_path, rule):
    """Quadratic strain energy: every rule integrates it exactly, so the
    tip deflection is gauss2's to round-off, and the fibre stress field
    has one row per station."""
    ref = _solver(_case(tmp_path, "gauss2"))
    ref.run(output_every=1, progress_every=10 ** 9)
    s = _solver(_case(tmp_path, rule))
    assert len(s.gauss_s) == int(rule[-1])
    s.run(output_every=1, progress_every=10 ** 9)
    assert s.monitors["w_tip"].values[-1] == pytest.approx(ref.monitors["w_tip"].values[-1], rel=1e-6)
    assert s.fiber_stress_rows().shape == (4, int(rule[-1]), 32)
    assert f"{rule[-1]}pt lobatto" in s.summary()


def test_the_kernel_matches_the_numpy_path_at_five_stations(tmp_path):
    pytest.importorskip("numba")
    conc = "{type: concrete02, fpc: -30.0e6, epsc0: -0.002, fcu: -6.0e6, epscu: -0.006, rat: 0.1, ft: 3.0e6, Ets: 1.5e9, rho: 2400}"
    bars = ",\n       bars: [{y: 0.15, z: 0.05, section: bar}, {y: -0.15, z: -0.05, section: bar}]"
    path = _case(tmp_path, "lobatto5", concrete=conc, bars=bars)
    s = _solver(path)
    assert s._kernel is not None, s.kernel_status
    assert s._state_c.shape[1] == 5 and s._fstress.shape == (4, 5, 34)
    rng = np.random.default_rng(5)
    s.q[:] = 2e-3 * rng.standard_normal(s.ndof)
    s.trial_mode = True                  # both evaluations from the same committed history
    fk, ak = s._internal_force_kernel()
    ft, at = s._internal_force_true()
    assert np.allclose(fk, ft, rtol=1e-10, atol=1e-8 * np.abs(ft).max())
    assert np.allclose(s.fiber_stress_rows(), s._fiber_stress_rows, rtol=1e-10, atol=1.0)


def test_the_sweep_gives_every_station_its_own_slab(tmp_path):
    from vfife.sweep import Sweep
    path = _case(tmp_path, "lobatto5")
    run_case(path, progress_every=10 ** 9)
    sw = Sweep(path)
    assert np.allclose(sw.slices, [0.0, 0.5 * (1 - np.sqrt(3 / 7)) / 2 + 0.0, 0.5 * (0.5 + 0.5 * (1 - np.sqrt(3 / 7))),
                                   0.5 * (0.5 + 0.5 * (1 + np.sqrt(3 / 7))), 0.5 * (1 + 0.5 * (1 + np.sqrt(3 / 7))), 1.0])
    pts, hexa, owner, cols, stat = sw.solid_frame(np.zeros((5, 3)), np.zeros((5, 3)))
    assert hexa.shape[0] == 4 * 32 * 5 and sorted(set(stat)) == [0, 1, 2, 3, 4]


def test_an_unknown_rule_is_refused_with_the_list(tmp_path):
    with pytest.raises(Exception, match="lobatto5|fiber_integration|integration"):
        prepare_case(_case(tmp_path, "lobatto7"))
