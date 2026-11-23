"""The fibre beam's staged ramp-in: the mechanism BeamSolver
has had since v0.1, opt-in here because it was measured to buy nothing on
the two shipped fibre cases and to cost up to 2.4x.

`stage_ramp_iterations: N` blends a prescribed displacement from where the
level finds it to the level's target over the first N iterations and forbids
convergence inside them. Without it the level may converge from iteration 2
on (never 1: the 0/0 of v0.1).
"""
import pathlib

import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.fiber_beam_solver import FiberBeamSolver


def _case(td, ramp=None, stages=3, budget=4000):
    msh = pathlib.Path(td) / "m.msh"
    n = 4
    nodes = "\n".join(f"{i+1} {i*0.25:.6f} 0 0" for i in range(n + 1))
    elems = "\n".join(f"{e+1} 1 2 1 1 {e+1} {e+2}" for e in range(n))
    msh.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{n+1}\n{nodes}\n"
                   f"$EndNodes\n$Elements\n{n}\n{elems}\n$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(td) / "c.yaml"
    yp.write_text(f"""
name: ramp
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
amplitudes: {{ramp: [[0, 0], [1, 1]]}}
materials:
  s: {{type: steel02, E: 2.0e11, Fy: 3.0e8, b: 0.01, rho: 7850}}
sections:
  m: {{type: fiber, material: s, shape: tube, diameter: 0.1, t: 0.005, nt: 12}}
assign: [{{region: all, section: m, orientation: [0, 0, 1]}}]
regions:
  base: {{box: [-0.01, 0.01, -1, 1, -1, 1]}}
  tip:  {{box: [0.99, 1.01, -1, 1, -1, 1]}}
constraints:
  - {{region: base, dofs: [1,1,1,1,1,1]}}
  - {{region: tip, dofs: [0,1,0,0,0,0], value: [0, 0.02, 0, 0, 0, 0], amplitude: ramp}}
solver:
  backend: numpy
  mode: relaxation
  load_steps: {stages}
  max_steps_per_load: {budget}
  tolerance: 1.0e-3
  kinetic_tolerance: 1.0e-6
  n_steps: 50{'' if ramp is None else f'''
  stage_ramp_iterations: {ramp}'''}
output:
  path: out.h5
  monitors:
    v: {{kind: displacement, region: tip, dof: y}}
    R: {{kind: reaction, region: base, dof: y}}
""", encoding="utf-8")
    return str(yp)


def _run(path):
    prep = prepare_case(path)
    sol = FiberBeamSolver(prep.case.model_dump(mode="python"), prep.mesh, prep.material)
    sol.run(output_every=1, progress_every=10 ** 9)
    return sol


def test_the_default_ramp_is_one_iteration(tmp_path):
    """The measured default: no ramp, and so no cost on a level that
    converges in a few dozen iterations."""
    off = _run(_case(tmp_path))
    one = _run(_case(tmp_path, ramp=1))
    assert off.step == one.step
    assert np.array_equal(off.monitors["R"].values, one.monitors["R"].values)


def test_a_ramp_forbids_convergence_inside_it_and_changes_nothing_after(tmp_path):
    off = _run(_case(tmp_path))
    ramped = _run(_case(tmp_path, ramp=1000, stages=3))
    # every level ran at least the ramp: 3 levels x 1000 iterations, more
    # than the unramped run needed for all three (it converges in ~560 each)
    assert ramped.step >= 3 * 1000 > off.step
    # and the levels' equilibria are the same equilibria
    # the same equilibria to the 1e-3 residual tolerance the levels converge to
    # (the tube yields at level 2; the path inside a level is not physical)
    assert np.allclose(ramped.monitors["R"].values, off.monitors["R"].values, rtol=5e-3)
    assert np.allclose(ramped.monitors["v"].values, off.monitors["v"].values, rtol=1e-6)


def test_the_prescribed_dof_is_blended_in_over_the_ramp(tmp_path, monkeypatch):
    """Inside the ramp `_apply_bc` receives approach < 1 and puts the driven
    dof between where the level began and its target; after the ramp it is
    at the target. Recorded through the base-class call itself."""
    from vfife.core import fiber_beam_solver as fbs
    seen = []
    original = fbs.FiberBeamSolver._apply_bc

    def spy(self, fixed, moving, prescribed, progress, start=None, approach=1.0):
        original(self, fixed, moving, prescribed, progress, start, approach)
        if prescribed:
            seen.append((progress, approach, float(self.q[prescribed[0][0][0]])))

    monkeypatch.setattr(fbs.FiberBeamSolver, "_apply_bc", spy)
    _run(_case(tmp_path, ramp=50, stages=2))
    level2 = [(a, q) for p, a, q in seen if p == 1.0]
    inside = [q for a, q in level2 if a < 1.0]
    assert 0 < len(inside) < 50 or len(inside) == 49
    # blended: strictly between the first level's 10 mm and the second's 20 mm
    assert all(0.01 < q < 0.02 for q in inside[:-1])
    # and at the target once the ramp is over
    assert all(q == pytest.approx(0.02) for a, q in level2 if a >= 1.0)
