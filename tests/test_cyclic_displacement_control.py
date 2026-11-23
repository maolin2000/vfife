"""A cyclic run that returns to zero displacement must be able to converge.

THE STATE IT FAILED AT WAS EXACTLY RIGHT. A displacement-controlled cyclic
protocol passes through zero between peaks. At such a stage there is no
external load at all, the internal force is round-off, and a residual
normalised by `max(|F_ext[free]|, |F_int|)` is that round-off divided by
itself: 0.998, at a state whose worst unbalanced force is 5e-11 N and whose
displacements are zero to four decimals in millimetres. The run then stops
with a convergence failure on the one stage it had solved perfectly
(found on Cai et al.'s Tube A).

The library's other cyclic cases never saw it because they all carry a second
load -- case_11 holds 819 kN of axial force while it pushes -- and that load
sets the scale. A pure displacement-controlled test does not.

The floor is a millionth of the largest force the analysis has had to carry,
which is inert wherever anything is loaded and is a SCALE rather than a number
of newtons: this library refuses to convert units, so a criterion written in
newtons would mean something different in a mm/tonne/MPa case.
"""

import pathlib
import tempfile

import numpy as np
import pytest

from vfife.core.diagnostics import ConvergenceFailure


def _case(td, peaks=(1.0e-3, 2.0e-3), load_steps=8, name="cyc"):
    """A short elastic cantilever pushed and pulled back to zero, nothing else."""
    msh = pathlib.Path(td) / f"{name}.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n"
                   "1 0 0 0\n2 0.25 0 0\n3 0.5 0 0\n$EndNodes\n$Elements\n2\n"
                   "1 1 2 1 1 1 2\n2 1 2 1 1 2 3\n$EndElements\n", encoding="utf-8")
    peak_list = ", ".join(f"{p:.8f}" for p in peaks)
    yp = pathlib.Path(td) / f"{name}.yaml"
    yp.write_text(f"""
name: {name}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
amplitudes:
  protocol: {{cyclic: {{peaks: [{peak_list}], repeats: 1}}}}
materials:
  s: {{type: steel_cyclic, grade: Q345B, E: 2.06e11, rho: 7850.0}}
sections:
  m: {{type: fiber, material: s, shape: tube, diameter: 0.048, t: 0.003,
       nt: 12}}
assign: [{{region: all, section: m, orientation: [0, 0, 1]}}]
regions:
  base: {{box: [-0.01, 0.01, -1, 1, -1, 1]}}
  tip: {{box: [0.49, 0.51, -1, 1, -1, 1]}}
constraints:
  - {{region: base, dofs: [1,1,1,1,1,1]}}
  - {{region: tip, dofs: [0,0,1,0,0,0], value: [0, 0, 1.0],
      amplitude: protocol}}
solver:
  backend: numpy
  mode: relaxation
  n_steps: 4000
  load_steps: {load_steps}
  tolerance: 1.0e-5
output:
  monitors:
    V: {{kind: reaction, region: base, dof: z}}
    w: {{kind: displacement, region: tip, dof: z}}
""", encoding="utf-8")
    return str(yp)


def _run(yaml_path):
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver

    prep = prepare_case(yaml_path)
    p = _with_element(prep.case, prep.element, prep.case_path)
    s = make_structural_solver(p, prep.mesh, prep.material)
    s.run(output_every=1, progress_every=10 ** 9)
    return s


def test_a_stage_at_zero_displacement_converges():
    """`load_steps = 8` with two peaks puts a stage exactly on every point of
    the protocol table, zeros included. Before the floor, the third stage --
    back at zero, nothing loaded, nothing moved -- raised ConvergenceFailure."""
    with tempfile.TemporaryDirectory() as td:
        s = _run(_case(td))
        w = np.array(s.monitors["w"].values)
        V = np.array(s.monitors["V"].values)
        assert len(w) == 8
        # the protocol is [+p, 0, -p, 0] per peak, and the run followed it
        assert w[0] == pytest.approx(1.0e-3, rel=1e-3)
        assert abs(w[1]) < 1e-9 and abs(w[3]) < 1e-9
        assert w[2] == pytest.approx(-1.0e-3, rel=1e-3)
        assert w[4] == pytest.approx(2.0e-3, rel=1e-3)
        # and the reaction came with it. THE FIRST STAGE IS PART OF THE TEST:
        # it used to read 0 N at 1 mm of tip displacement, because `fint` in
        # the convergence check is one iteration stale and on iteration 1 of a
        # stage that is the previous equilibrium -- zero against zero under
        # pure displacement control, so the stage converged before the
        # prescribed displacement had done anything.
        k = [abs(v / d) for v, d in zip(V, w) if abs(d) > 1e-12]
        assert len(k) == 4
        assert max(k) - min(k) < 1e-6 * max(k), f"not one straight line: {k}"
        assert V[0] * V[2] < 0, "the tip was pushed both ways"
        for i in (1, 3, 5, 7):
            assert abs(V[i]) < 1e-6 * abs(V[0]), "no load at the zero crossing"


def test_the_floor_is_inert_where_anything_is_loaded():
    """The guard may not loosen the criterion where it was doing its job: at a
    loaded stage the scale is still the current force, so a run that would
    have failed still fails. Driven here by asking for a tolerance no
    relaxation can reach."""
    with tempfile.TemporaryDirectory() as td:
        yp = _case(td, peaks=(2.0e-3,), load_steps=4, name="tight")
        text = pathlib.Path(yp).read_text(encoding="utf-8").replace("tolerance: 1.0e-5",
                                                    "tolerance: 1.0e-16")
        pathlib.Path(yp).write_text(text, encoding="utf-8")
        with pytest.raises(ConvergenceFailure) as excinfo:
            _run(yp)
        # it fails at a LOADED stage (1 or 3), not at a zero crossing (2, 4),
        # which is the whole point: the floor did not make the criterion
        # unreachable, it made the degenerate state reachable
        assert "stage 3/4" in str(excinfo.value) or "stage 1/4" in str(excinfo.value)
        resid = float(str(excinfo.value).split("residual=")[1].split(" ")[0])
        assert resid < 1e-9, "it stopped short of an impossible tolerance, not at 0/0"
