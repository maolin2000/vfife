"""A rupture latch inside a staged relaxation.

THE MECHANISM, on the smallest model that has it. A bar of four fibre members
is pulled at its free end by a prescribed displacement, one load stage at a
time. A stage imposes its whole increment on the end node at once, so on the
first iterations the LAST member carries all of it: strain u/h, four times
the strain the converged bar will have. That is the fictitious dynamics of the
relaxation, not a state of the structure -- but a law that cut its fibre on
the call that crossed the rupture strain cut it right there, the member
stopped carrying, the rest of the bar never moved, and the stage converged on
a broken bar that the converged strain had never broken.

On a beam the same thing happens at the elements next to the load points, and
it is worse the shorter the element (the transient strain goes as the
increment over h). Li's A7 corroded beam broke at 19.7 kN m with 40 mm
elements and 29.5 with 80 mm ones -- read in v0.4 as a mesh-dependent
rupture criterion. With the latch read against the ACCEPTED state, both
meshes reach the rupture stress at 39.5 kN m, the section-method value.

WHAT IS ASSERTED
  1. a stage whose converged strain is below the limit ends intact, however
     far the transient overshot it (the old code: reaction 0);
  2. a stage whose converged strain passes the limit is re-solved with the
     bar cut, so what it records is an equilibrium (reaction ~0), not the
     intact positions paired with the cut forces;
  3. the stage before it still carries the full elastic force.
"""
import pathlib
import tempfile

import numpy as np
import pytest

E, A = 2.0e11, 2.5e-3          # the 0.05 x 0.05 steel section below
RUP = 2.0e-3


def _case(td, steps, eps_end, n_elem=4, ramp=None):
    L = 1.0
    msh = pathlib.Path(td) / "bar.msh"
    nodes = "\n".join(f"{i+1} {i*L/n_elem:.6f} 0 0" for i in range(n_elem + 1))
    elems = "\n".join(f"{i+1} 1 2 1 1 {i+1} {i+2}" for i in range(n_elem))
    msh.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{n_elem+1}\n"
                   f"{nodes}\n$EndNodes\n$Elements\n{n_elem}\n{elems}\n$EndElements\n", encoding="utf-8")
    ramp_line = f"\n  stage_ramp_iterations: {ramp}" if ramp else ""
    yp = pathlib.Path(td) / "bar.yaml"
    yp.write_text(f"""
name: bar
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
materials:
  steel: {{type: steel02, E: {E}, Fy: 1.0e9, b: 0.0, R0: 20.0, rho: 7850.0,
           rupture_strain: {RUP}}}
sections:
  s: {{type: fiber, material: steel, shape: rect, width: 0.05, height: 0.05,
       ny: 2, nz: 2}}
assign: [{{region: all, section: s}}]
regions:
  base: {{box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}}
  tip: {{box: [{L - 0.01}, {L + 0.01}, -0.01, 0.01, -0.01, 0.01]}}
constraints:
- {{region: all, dofs: [0, 1, 1, 1, 1, 1]}}
- {{region: base, dofs: [1, 1, 1, 1, 1, 1]}}
steps:
- name: pull
  load_steps: {steps}
  constraints:
  - {{region: tip, dofs: [1, 0, 0, 0, 0, 0], value: {eps_end * L}}}
solver:
  backend: numpy
  mode: relaxation
  n_steps: 100
  max_steps_per_load: 50000
  tolerance: 1.0e-6
  kinetic_tolerance: 1.0e-10{ramp_line}
  damping: {{type: kinetic}}
output:
  path: {str(pathlib.Path(td) / "bar.h5").replace(chr(92), "/")}
  every: 1
  fields: [displacement]
  monitors:
    R: {{kind: reaction, region: tip, dof: x}}
""", encoding="utf-8")
    return str(yp)


def _reactions(yp):
    from vfife.api import run_case
    out = run_case(yp)
    assert out.status in ("COMPLETED", "CONVERGED", "SUCCESS") or out.ok, out.message
    return np.asarray(out.monitors["R"])


def test_a_stage_that_converges_below_the_limit_ends_intact():
    """One stage to 0.7 of the rupture strain. The first iterations put four
    times that into the last member; the converged bar is at 0.7 and whole."""
    with tempfile.TemporaryDirectory() as td:
        R = _reactions(_case(td, steps=1, eps_end=0.7 * RUP))
    assert abs(R[-1]) == pytest.approx(E * A * 0.7 * RUP, rel=1e-3)


def test_the_stage_that_breaks_the_bar_records_the_broken_equilibrium():
    """Four stages to 1.2 of the limit: 0.3, 0.6, 0.9 carry, 1.2 breaks, and
    the frame recorded for 1.2 is the bar after it broke -- no force."""
    with tempfile.TemporaryDirectory() as td:
        R = np.abs(_reactions(_case(td, steps=4, eps_end=1.2 * RUP)))
    carried = R[-2]
    assert carried == pytest.approx(E * A * 0.9 * RUP, rel=1e-3)
    assert R[-1] < 1e-6 * carried


def _series_case(td, steps, eps_end):
    """Four members in series; the one at the base has a strand that breaks,
    the other three a strand that does not. Every member also has elastic
    fibres, so a broken member still carries -- and the chain has to find a
    new equilibrium, which is what the stage must record."""
    L, n = 1.0, 4
    msh = pathlib.Path(td) / "chain.msh"
    nodes = "\n".join(f"{i+1} {i*L/n:.6f} 0 0" for i in range(n + 1))
    elems = "\n".join(f"{i+1} 1 2 {2 if i == 0 else 1} 1 {i+1} {i+2}" for i in range(n))
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
                   "$PhysicalNames\n2\n1 1 \"strong\"\n1 2 \"weak\"\n$EndPhysicalNames\n"
                   f"$Nodes\n{n+1}\n{nodes}\n$EndNodes\n"
                   f"$Elements\n{n}\n{elems}\n$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(td) / "chain.yaml"
    yp.write_text(f"""
name: chain
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
materials:
  body: {{type: linear_elastic, E: 3.0e10, nu: 0.2, rho: 2400.0}}
  brittle: {{type: steel02, E: {E}, Fy: 1.0e9, b: 0.0, R0: 20.0, rho: 7850.0,
             rupture_strain: {RUP}}}
  tough: {{type: steel02, E: {E}, Fy: 1.0e9, b: 0.0, R0: 20.0, rho: 7850.0}}
sections:
  pb: {{type: bar, material: brittle, area: 2.0e-4}}
  pt: {{type: bar, material: tough, area: 2.0e-4}}
  weak: {{type: fiber, material: body, shape: rect, width: 0.05, height: 0.05,
          ny: 2, nz: 2, bars: [{{y: 0.0, z: 0.01, section: pb}}, {{y: 0.0, z: -0.01, section: pb}}]}}
  strong: {{type: fiber, material: body, shape: rect, width: 0.05, height: 0.05,
            ny: 2, nz: 2, bars: [{{y: 0.0, z: 0.01, section: pt}}, {{y: 0.0, z: -0.01, section: pt}}]}}
assign: [{{region: strong, section: strong}}, {{region: weak, section: weak}}]
regions:
  base: {{box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}}
  tip: {{box: [{L - 0.01}, {L + 0.01}, -0.01, 0.01, -0.01, 0.01]}}
constraints:
- {{region: all, dofs: [0, 1, 1, 1, 1, 1]}}
- {{region: base, dofs: [1, 1, 1, 1, 1, 1]}}
steps:
- name: pull
  load_steps: {steps}
  constraints:
  - {{region: tip, dofs: [1, 0, 0, 0, 0, 0], value: {eps_end * L}}}
solver:
  backend: numpy
  mode: relaxation
  n_steps: 100
  max_steps_per_load: 50000
  tolerance: 1.0e-7
  kinetic_tolerance: 1.0e-12
  damping: {{type: kinetic}}
output:
  path: {str(pathlib.Path(td) / "chain.h5").replace(chr(92), "/")}
  every: 1
  fields: [displacement]
  monitors:
    R: {{kind: reaction, region: tip, dof: x}}
""", encoding="utf-8")
    return str(yp)


def test_the_stage_that_breaks_a_member_is_solved_again_before_it_is_recorded():
    """Two stages to 1.2 of the limit. At the second, the base member's strand
    breaks; the member keeps its elastic fibres, so the chain re-balances with
    a softer link: N = u / sum(h / EA). Recorded without that re-solve, the
    tip member at the intact positions still reads (EA_body + EA_strand) eps
    -- 27% too much, and not an equilibrium of anything."""
    ea_body, ea_strand, h = 3.0e10 * 2.5e-3, E * 4.0e-4, 0.25
    u = 1.2 * RUP
    with tempfile.TemporaryDirectory() as td:
        R = np.abs(_reactions(_series_case(td, steps=2, eps_end=1.2 * RUP)))
    assert R[-2] == pytest.approx((ea_body + ea_strand) * 0.6 * RUP, rel=2e-3)
    expected = u / (h / ea_body + 3 * h / (ea_body + ea_strand))
    assert R[-1] == pytest.approx(expected, rel=2e-3)


def test_262_263_a_restart_inside_the_level_a_strand_breaks_at_still_solves_it_again():
    """Stopped inside the level whose trial passes the limit, the checkpoint
    holds the trial fibres -- the strand's points already failed -- and the
    restarted level compared the commit with that count: nothing new failed,
    the level was not solved again, and its frame read the intact chain's
    force, 372 kN where the uninterrupted run reads 294 kN.
    The count is the accepted state's."""
    from vfife.api import run_case
    from vfife.core.diagnostics import AnalysisInterrupted
    with tempfile.TemporaryDirectory() as td:
        yp = _series_case(td, steps=4, eps_end=1.6 * RUP)
        import yaml
        d = yaml.safe_load(open(yp, encoding="utf-8"))
        box = {}
        d["output"]["path"] = str(pathlib.Path(td) / "w.h5")
        w = run_case(d, on_ready=lambda s, p: box.setdefault("w", s))
        calls = {"n": 0}

        def stop(*_a):
            calls["n"] += 1
            return calls["n"] < 110                # inside the third level, past the limit

        d["output"]["path"] = str(pathlib.Path(td) / "p.h5")
        with pytest.raises(AnalysisInterrupted):
            run_case(d, on_progress=stop, progress_every=200)
        d["output"]["path"] = str(pathlib.Path(td) / "r.h5")
        r = run_case(d, restart=str(pathlib.Path(td) / "p.h5"), on_ready=lambda s, p: box.setdefault("r", s))
        # ... and a new run that starts from that state (`initial_state`),
        # one level to 1.2 of the limit: the strand breaks at its commit
        d2 = yaml.safe_load(yaml.safe_dump(d))
        d2["output"]["path"] = str(pathlib.Path(td) / "i.h5")
        d2["solver"]["initial_state"] = str(pathlib.Path(td) / "p.h5")
        d2["steps"][0]["load_steps"] = 1
        d2["steps"][0]["constraints"][0]["value"] = 1.2 * RUP
        i = run_case(d2)
    Rw, Rr = np.asarray(w.monitors["R"]), np.asarray(r.monitors["R"])
    assert np.array_equal(Rr, Rw[-len(Rr):])
    assert np.array_equal(box["r"].q, box["w"].q)
    ea_body, ea_strand, h = 3.0e10 * 2.5e-3, E * 4.0e-4, 0.25
    expected = 1.2 * RUP / (h / ea_body + 3 * h / (ea_body + ea_strand))
    assert abs(np.asarray(i.monitors["R"])[-1]) == pytest.approx(expected, rel=2e-3)
