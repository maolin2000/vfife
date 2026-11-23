"""A hollow member that yields and then buckles: three features at once.

This is the only place in the library where a HOLLOW fibre section, the
CYCLIC steel law and SECOND-ORDER geometry all have to be right in the same
answer, and each of them can be wrong in a way the others hide: a section
area error moves the squash load, a law that leaves its elastic branch too
late moves it the same way, and an imperfection in the wrong plane moves it
the other way by about as much. Version 0.1 found two defects in the law
through exactly this number.

The member is the one Cai et al. (2023) call Tube A -- RHS 25 x 50 x 2.5,
L = 1100 mm, f_y = 288 MPa, camber l/500, built in at one end and on a roller
pin at the other -- because it is a published member whose squash load is
A f_y to three figures, not because this library keeps that paper's case.

Both tests are rate-free: staged relaxation, not an explicit drive.
"""

import pathlib
import tempfile

import numpy as np
import pytest


L, E, FY, FU = 1.100, 215.15e9, 288.0e6, 361.0e6
B, H, T = 0.025, 0.050, 0.0025
CAMBER = L / 500
N_ELEM = 8

MATERIAL = f"""  s: {{type: steel_cyclic, E: {E}, rho: 7850.0, fy: {FY}, fu: {FU},
       f_uf: {FY}, k1: 5.0, k2: 46.0, k3: {FU / FY:.6f}, a: 5.796, b0: 2.486,
       b1: 0.608, b2: -0.00046, eps_cri: 0.43, lambda_dsps: 0.23}}"""


def _case(td, umax, axis, stages):
    """`axis` is the direction of the l/500 half-sine camber: "z" bends the
    member about its strong axis, "y" about its weak one."""
    msh = pathlib.Path(td) / "tube.msh"

    def node(i):
        s = CAMBER * np.sin(np.pi * i / N_ELEM)
        y, z = (s, 0.0) if axis == "y" else (0.0, s)
        return f"{i+1} {i*L/N_ELEM:.6f} {y:.6f} {z:.6f}"

    nodes = "\n".join(node(i) for i in range(N_ELEM + 1))
    elems = "\n".join(f"{i+1} 1 2 1 1 {i+1} {i+2}" for i in range(N_ELEM))
    msh.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{N_ELEM+1}\n"
                   f"{nodes}\n$EndNodes\n$Elements\n{N_ELEM}\n{elems}\n"
                   f"$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(td) / "tube.yaml"
    yp.write_text(f"""
name: hollow_member_push
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
amplitudes: {{ramp: [[0, 0], [1, 1]]}}
materials:
{MATERIAL}
sections:
  m: {{type: fiber, material: s, shape: box, width: {B}, height: {H}, t: {T},
       ny: 4, nz: 8, integration: lobatto3}}
assign: [{{region: all, section: m, orientation: [0, 1, 0]}}]
regions:
  fixed: {{box: [-0.01, 0.01, -1, 1, -1, 1]}}
  loaded: {{box: [{L-0.01}, {L+0.01}, -1, 1, -1, 1]}}
constraints:
  - {{region: fixed, dofs: [1,1,1,1,1,1]}}
  - {{region: loaded, dofs: [1,1,1,0,0,0], value: [-{umax}, 0, 0, 0, 0, 0],
      amplitude: ramp}}
solver:
  backend: numpy
  mode: relaxation
  n_steps: 30000
  load_steps: {stages}
  tolerance: 2.0e-3
  max_steps_per_load: 30000
output:
  monitors:
    N: {{kind: reaction, region: fixed, dof: x}}
    u: {{kind: displacement, region: loaded, dof: x}}
""", encoding="utf-8")
    return str(yp)


def _push(axis, umax=2.4e-3, stages=12):
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver

    with tempfile.TemporaryDirectory() as td:
        prep = prepare_case(_case(td, umax, axis, stages))
        p = _with_element(prep.case, prep.element, prep.case_path)
        s = make_structural_solver(p, prep.mesh, prep.material)
        s.run(output_every=1, progress_every=10 ** 9)
        N = np.abs(np.array(s.monitors["N"].values)) / 1e3
        u = np.abs(np.array(s.monitors["u"].values)) * 1e3
        return N, u, float(s.total_area[0])


def test_the_member_reaches_the_squash_load_and_then_turns_over():
    """It gets close to A f_y, at the right shortening, and then softens.

    91.8 kN at 1.40 mm here against a squash load of 100.8. The shortfall is
    the second-order amplification of the l/500 camber: the strong-axis Euler
    load at 0.7L is 389 kN, so 1/(1 - P/P_cr) = 1.30. Version 0.1 guessed
    that the published member's end stiffeners would recover most of it; the
    test below, written once several sections could share a model,
    measures 2.6% -- the peak is set by the mid-span hinge
    yielding under N + M, which stiffer ends barely touch. So the band here
    is centred on what the member does, stiffened or not.

    The three things being guarded, and how each would break it: the box
    section's area (the squash load itself), the cyclic law's elastic exit
    (a fibre that overshoots f_y lifts the peak -- it once reached 14 f_u),
    and the geometry (a linear member would not turn over at all).
    """
    N, u, area = _push("z")
    assert area == pytest.approx(350e-6, rel=2e-3), "RHS 25x50x2.5 is 350 mm2"
    squash = area * FY / 1e3
    peak = N.max()
    at = u[int(np.argmax(N))]
    assert 0.86 * squash < peak < 0.98 * squash, (
        f"peak {peak:.1f} kN against a squash load of {squash:.1f}")
    assert 1.2 < at < 1.7, f"peak at {at:.2f} mm; f_y L / E is 1.47"
    # and it TURNS OVER: this is a buckling peak, not the end of the ramp
    assert N[-1] < 0.93 * peak, "no post-peak degradation -- it never buckled"


def test_the_imperfection_plane_is_what_decides_the_peak():
    """The same member with the same camber, bent the other way, is 20% weaker.

    An earlier pass put the camber in y on the argument that a member buckles
    the weak way. It does when it is free to; this one is not. The two models
    differ by one coordinate and produce answers 20% apart, which is why it
    is asserted rather than remembered.
    """
    strong, _, _ = _push("z")
    weak, uw, _ = _push("y")
    assert weak.max() < 0.85 * strong.max()
    assert uw[int(np.argmax(weak))] < 1.3, "the weak plane also peaks earlier"


def _case_stiffened(td, umax, t_end, stages=12):
    """The same member with Fig. 17's `element with stiffener` at both ends,
    modelled as the first and last element carrying a thicker wall."""
    p = pathlib.Path(_case(td, umax, "z", stages))
    text = p.read_text(encoding="utf-8")
    dx = L / N_ELEM
    old = f"""sections:
  m: {{type: fiber, material: s, shape: box, width: {B}, height: {H}, t: {T},
       ny: 4, nz: 8, integration: lobatto3}}
assign: [{{region: all, section: m, orientation: [0, 1, 0]}}]
regions:"""
    new = f"""sections:
  m:  {{type: fiber, material: s, shape: box, width: {B}, height: {H}, t: {T},
        ny: 4, nz: 8, integration: lobatto3}}
  st: {{type: fiber, material: s, shape: box, width: {B}, height: {H}, t: {t_end},
        ny: 4, nz: 8, integration: lobatto3}}
assign:
  - {{region: mid,  section: m,  orientation: [0, 1, 0]}}
  - {{region: ends, section: st, orientation: [0, 1, 0]}}
regions:
  mid:   {{box: [{dx-0.001}, {L-dx+0.001}, -1, 1, -1, 1]}}
  ends:  {{union: [end_a, end_b]}}
  end_a: {{box: [-0.01, {dx+0.001}, -1, 1, -1, 1]}}
  end_b: {{box: [{L-dx-0.001}, {L+0.01}, -1, 1, -1, 1]}}"""
    assert old in text
    p.write_text(text.replace(old, new), encoding="utf-8")
    return str(p)


def test_stiffened_ends_raise_the_peak_a_little_and_bring_it_forward():
    """Fig. 17's end stiffeners, now that a model can carry two sections.

    Measured: plain 90.99 kN at 1.40 mm; 10 mm end walls 93.37 kN at 1.20 mm.
    +2.6%, and earlier because the member is axially stiffer. NOT the 12%
    v0.1 estimated from the elastic amplification alone -- the peak
    is an inelastic one. Asserted as a direction and a modest band, so that
    the multi-section machinery is exercised on the member that asked for it
    and the correction to v0.1 is on record rather than in a note.
    """
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver

    def push(path):
        prep = prepare_case(path)
        p = _with_element(prep.case, prep.element, prep.case_path)
        s = make_structural_solver(p, prep.mesh, prep.material)
        s.run(output_every=1, progress_every=10 ** 9)
        N = np.abs(np.array(s.monitors["N"].values)) / 1e3
        u = np.abs(np.array(s.monitors["u"].values)) * 1e3
        return N.max(), u[int(np.argmax(N))], s

    with tempfile.TemporaryDirectory() as td:
        plain, at_plain, _ = push(_case(td, 2.4e-3, "z", 12))
        stiff, at_stiff, sol = push(_case_stiffened(td, 2.4e-3, 0.010))
    assert sol._section_of is not None and sol.fiber_valid.shape[1] == sol.n_fibers
    assert 1.01 * plain < stiff < 1.06 * plain, (plain, stiff)
    assert at_stiff <= at_plain, (at_plain, at_stiff)
