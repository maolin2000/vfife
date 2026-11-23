"""Cables: unstressed length, prestress and slack on truss members.

A truss member in a `mixed_line` model has an unstressed length L_u -- its
mesh length unless the assignment states `prestress` (L_u = L0/(1+sigma/E))
or `unstressed_length` (the group's total, shared over its elements in
proportion to their mesh lengths). Its force is E A (L - L_u)/L_u, clipped
at zero when `tension_only`. A standalone truss2 takes `prestress` and
`tension_only` through the strain offset the embedded-tendon path already
had. Plain trusses are untouched: same forces, same fictitious masses.

The physics anchor is Irvine's elastic catenary, cases/case_22 in full
(40 elements, 45 s) and a ten-element version here.
"""
import pathlib
import sys

import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.structural import make_structural_solver
from vfife.api import _with_element

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))


def _line_mesh(td, pts, groups):
    """A polyline mesh; `groups` maps a physical name to its element indices."""
    msh = pathlib.Path(td) / "m.msh"
    names = list(groups)
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names) + 2),
           '0 101 "first"', '0 102 "last"']
    out += [f'1 {k + 1} "{n}"' for k, n in enumerate(names)]
    out += ["$EndPhysicalNames", "$Nodes", str(len(pts))]
    out += [f"{k + 1} {x:.10g} {y:.10g} {z:.10g}" for k, (x, y, z) in enumerate(pts)]
    out += ["$EndNodes", "$Elements", str(len(pts) - 1 + 2), "1 15 2 101 101 1",
            f"2 15 2 102 102 {len(pts)}"]
    e = 3
    for k, n in enumerate(names):
        for i in groups[n]:
            out.append(f"{e} 1 2 {k + 1} {k + 1} {i + 1} {i + 2}"); e += 1
    out += ["$EndElements", ""]
    msh.write_text("\n".join(out), encoding="utf-8")
    return msh


def _solver(path):
    prep = prepare_case(path)
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


def _two_bar_case(td, assign, E=2.0e11, extra=""):
    msh = _line_mesh(td, [(0, 0, 0), (1, 0, 0), (2, 0, 0)], {"cable": [0, 1]})
    yp = pathlib.Path(td) / "c.yaml"
    yp.write_text(f"""
name: two_bar
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  s: {{type: linear_elastic, E: {E}, nu: 0.3, rho: 7850}}
sections:
  bar:  {{type: bar, material: s, shape: general, area: 1.0e-4}}
  stub: {{type: beam, material: s, shape: general, area: 1.0e-4, iy: 1e-8, iz: 1e-8, j: 2e-8}}
assign:
{assign}
regions:
  ends: {{union: [first, last]}}
{extra}constraints:
  - {{region: ends, dofs: [1,1,1,1,1,1]}}
  - {{region: all, dofs: [0,1,1,1,1,1]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, max_steps_per_load: 5000,
          tolerance: 1.0e-8, kinetic_tolerance: 1.0e-10, n_steps: 50}}
output: {{path: out.h5, monitors: {{R: {{kind: reaction, region: first, dof: x}}}}}}
""", encoding="utf-8")
    return str(yp)


def test_prestress_and_unstressed_length_are_the_same_statement(tmp_path):
    """L_u = L0 / (1 + sigma/E): a group stated either way has the same
    unstressed lengths, element by element, and the same force at u = 0."""
    E, sigma = 2.0e11, 4.0e8
    a = _solver(_two_bar_case(tmp_path, f"  - {{region: cable, section: bar, prestress: {sigma}}}", E))
    Lu = 2.0 / (1 + sigma / E)
    b = _solver(_two_bar_case(tmp_path, f"  - {{region: cable, section: bar, unstressed_length: {Lu}}}", E))
    assert np.allclose(a.member_Lu, b.member_Lu, rtol=1e-14) and a.has_cables and b.has_cables
    assert np.allclose(a.member_Lu, [Lu / 2, Lu / 2])
    fa, _ = a.internal_force(); fb, _ = b.internal_force()
    assert np.allclose(fa, fb, rtol=1e-12)
    # at u = 0 the bar carries sigma A: the end nodes see +-40 kN
    assert abs(fa[0]) == pytest.approx(sigma * 1.0e-4, rel=1e-9)


def test_a_tension_only_cable_carries_no_compression(tmp_path):
    """The same two bars, shortened by prescribing the far end inward: a
    plain truss pushes back, a cable goes slack and pushes nothing."""
    def run(flag):
        path = _two_bar_case(
            tmp_path, f"  - {{region: cable, section: bar{flag}}}",
            extra="")
        s = _solver(path)
        # shorten by 1 mm: prescribe the last node 1 mm toward the first
        last = s._region("last")[0]
        s.q[s.dofs.node_dofs(last).ravel()[0]] = -1.0e-3
        f, actions = s.internal_force()
        return actions[:, 0], f
    n_truss, _ = run("")
    n_cable, f_cable = run(", tension_only: true")
    # only the second bar is shortened (its far node moved 1 mm over 1 m):
    # the plain truss carries E A eps = 20 kN of compression there
    assert np.allclose(n_truss, [0.0, 2.0e11 * 1.0e-4 * 1e-3])
    assert np.all(n_cable == 0.0) and np.all(f_cable == 0.0)


def test_a_plain_truss_is_untouched(tmp_path):
    """No cable flag: L_u is the mesh length, no slack, and the reference
    stiffness (hence the fictitious masses) is the axial one -- not the
    isotropic block a cable gets."""
    s = _solver(_two_bar_case(tmp_path, "  - {region: cable, section: bar}"))
    assert not s.has_cables and np.array_equal(s.member_Lu, s.length)
    K = s.K.toarray()
    # the middle node's transverse translations have no stiffness on a
    # straight plain truss; a cable's would carry the axial value
    mid = s._region("all")[1]
    d = s.dofs.node_dofs(mid).ravel()
    assert K[d[1], d[1]] == 0.0 and K[d[2], d[2]] == 0.0 and K[d[0], d[0]] > 0.0
    c = _solver(_two_bar_case(tmp_path, "  - {region: cable, section: bar, tension_only: true}"))
    Kc = c.K.toarray()
    assert Kc[d[1], d[1]] == Kc[d[0], d[0]] == Kc[d[2], d[2]] > 0.0


def test_a_standalone_truss_takes_prestress_and_tension_only(tmp_path):
    msh = _line_mesh(tmp_path, [(0, 0, 0), (1, 0, 0)], {"cable": [0]})
    for flag, expect in (("", -2.0e4), (", tension_only: true", 0.0)):
        yp = pathlib.Path(tmp_path) / f"s{len(flag)}.yaml"
        yp.write_text(f"""
name: one_bar
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  s: {{type: linear_elastic, E: 2.0e11, nu: 0.3, rho: 7850}}
sections:
  bar: {{type: bar, material: s, shape: general, area: 1.0e-4}}
assign:
  - {{region: all, section: bar, prestress: -2.0e8{flag}}}
regions: {{}}
constraints:
  - {{region: first, dofs: [1,1,1]}}
  - {{region: last, dofs: [1,1,1]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, n_steps: 50}}
output: {{path: out.h5, monitors: {{}}}}
""", encoding="utf-8")
        s = _solver(str(yp))
        assert s.pre.prestrain == pytest.approx(-1.0e-3)
        f, _sig = s._force()
        # a compressive prestress of 200 MPa on 1e-4 m^2: -20 kN, or nothing
        assert f[1, 0] == pytest.approx(expect, abs=1e-6)


def test_unstressed_length_needs_a_named_region(tmp_path):
    msh = _line_mesh(tmp_path, [(0, 0, 0), (1, 0, 0)], {"cable": [0]})
    yp = pathlib.Path(tmp_path) / "u.yaml"
    yp.write_text(f"""
name: one_bar
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  s: {{type: linear_elastic, E: 2.0e11, nu: 0.3, rho: 7850}}
sections:
  bar: {{type: bar, material: s, shape: general, area: 1.0e-4}}
assign:
  - {{region: all, section: bar, unstressed_length: 1.02}}
regions: {{}}
constraints:
  - {{region: first, dofs: [1,1,1]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, n_steps: 50}}
output: {{path: out.h5, monitors: {{}}}}
""", encoding="utf-8")
    with pytest.raises(ValueError, match="unstressed_length"):
        prepare_case(str(yp))


def test_a_ten_element_cable_finds_irvines_catenary(tmp_path):
    """The shipped case_22 with ten elements instead of forty: the funicular
    polygon of ten links sits within 0.5% of the smooth catenary."""
    from generate_cable_catenary import build
    msh = pathlib.Path(tmp_path) / "cat.msh"
    info = build(str(msh), n=10)
    yp = pathlib.Path(tmp_path) / "cat.yaml"
    yp.write_text(f"""
name: catenary_10
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  steel: {{type: linear_elastic, E: 2.0e10, nu: 0.3, rho: 7850.0}}
sections:
  main: {{type: bar, material: steel, shape: general, area: 0.05}}
assign:
  - {{region: cable, section: main, unstressed_length: 102.0, tension_only: true}}
regions:
  ends: {{union: [left, right]}}
constraints:
  - {{region: ends, dofs: [1, 1, 1, 1, 1, 1]}}
  - {{region: all, dofs: [0, 1, 0, 1, 1, 1]}}
loads:
  - {{region: all, gravity: [0.0, 0.0, -9.80665]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, max_steps_per_load: 60000,
          stage_ramp_iterations: 50, tolerance: 1.0e-6, kinetic_tolerance: 1.0e-9, n_steps: 100}}
output:
  path: out.h5
  monitors:
    H: {{kind: reaction, region: left, dof: x}}
    V: {{kind: reaction, region: left, dof: z}}
    sag: {{kind: displacement, region: mid, dof: z}}
""", encoding="utf-8")
    s = _solver(str(yp))
    s.run(output_every=1, progress_every=10 ** 9)
    H = -s.monitors["H"].values[-1]; V = s.monitors["V"].values[-1]; sag = -s.monitors["sag"].values[-1]
    assert H == pytest.approx(info["H"], rel=5e-3)
    assert V == pytest.approx(info["V"], rel=1e-4)
    assert sag == pytest.approx(info["sag_mid"], rel=5e-3)
    assert s.has_cables and np.all(s.end_actions[:, 6] > 0.0)      # every element in tension
