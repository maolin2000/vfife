"""Modal analysis from the finite-difference tangent and the physical mass:
K phi = omega^2 M phi at a state.

Anchors: the Euler beam's (n pi / L)^2 sqrt(EI/m); Irvine's cable modes
at the catenary state -- the antisymmetric in-plane modes of a shallow
cable are the taut string's even modes, 2 n pi / L sqrt(H / m), and the
first symmetric one solves tan(w/2) = w/2 - (4/lambda^2)(w/2)^3 -- which
only the tangent at the LOADED state can give, since a cable's transverse
stiffness is its tension.
"""
import pathlib
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

from vfife import modal
from vfife.api import _with_element, prepare_case
from vfife.core.structural import make_structural_solver

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))


def _solver(path):
    prep = prepare_case(path)
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


def _line(td, n, L, name="m"):
    msh = pathlib.Path(td) / f"{name}.msh"
    nodes = "\n".join(f"{i + 1} {i * L / n:.8f} 0 0" for i in range(n + 1))
    elems = "\n".join(f"{i + 1} 1 2 1 1 {i + 1} {i + 2}" for i in range(n))
    msh.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{n + 1}\n{nodes}\n$EndNodes\n"
                   f"$Elements\n{n}\n{elems}\n$EndElements\n", encoding="utf-8")
    return msh


def test_a_simply_supported_beam_has_the_euler_frequencies(tmp_path):
    L, E, I, A, rho = 10.0, 2.0e11, 1.0e-4, 0.01, 7850.0
    msh = _line(tmp_path, 20, L)
    yp = pathlib.Path(tmp_path) / "ss.yaml"
    yp.write_text(f"""
name: ss
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  s: {{type: linear_elastic, E: {E}, nu: 0.3, rho: {rho}}}
sections:
  b: {{type: beam, material: s, shape: general, area: {A}, iy: {I}, iz: {I}, j: {2 * I}}}
assign: [{{region: all, section: b}}]
regions:
  left: {{box: [-0.01, 0.01, -1, 1, -1, 1]}}
  right: {{box: [{L - 0.01}, {L + 0.01}, -1, 1, -1, 1]}}
constraints:
  - {{region: all, dofs: [0, 1, 0, 1, 0, 1]}}
  - {{region: left, dofs: [1, 1, 1, 1, 0, 1]}}
  - {{region: right, dofs: [0, 1, 1, 1, 0, 1]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, n_steps: 50}}
output: {{path: out.h5, monitors: {{}}}}
""", encoding="utf-8")
    s = _solver(str(yp))
    res = modal.analyse(s, n_modes=4)
    m = rho * A
    exact = [(k * np.pi / L) ** 2 * np.sqrt(E * I / m) / (2 * np.pi) for k in (1, 2, 3)]
    # the lumped mass (with its rotary share) is soft by 0.1 / 0.4 / 0.9 %
    assert res.frequencies[0] == pytest.approx(exact[0], rel=3e-3)
    assert res.frequencies[1] == pytest.approx(exact[1], rel=1e-2)
    assert res.frequencies[2] == pytest.approx(exact[2], rel=2e-2)
    # the first mode is vertical (global z) and carries 8/pi^2 of the mass
    assert res.participation[0, 2] > 0.8 and res.participation[0, 0] < 1e-6
    # shapes: unit amplitude, a half sine at the middle node, zero on the held dofs
    sh = res.shapes[0].reshape(21, 6)
    assert abs(sh[10, 2]) == pytest.approx(1.0) and np.allclose(sh[:, 1], 0.0) and np.allclose(sh[0, 2], 0.0)
    assert res.n_free == 60 and res.note == ""


def test_a_fibre_cantilever_first_mode(tmp_path):
    """1.875^2 / (2 pi L^2) sqrt(EI/m) on a fibre section (the tangent is
    the finite difference of the fibre integration)."""
    L, E, rho = 4.0, 2.0e11, 7850.0
    b, h = 0.2, 0.4
    msh = _line(tmp_path, 8, L)
    yp = pathlib.Path(tmp_path) / "cant.yaml"
    yp.write_text(f"""
name: cant
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  s: {{type: linear_elastic, E: {E}, nu: 0.0, rho: {rho}}}
sections:
  f: {{type: fiber, material: s, shape: rect, width: {h}, height: {b}, ny: 8, nz: 4}}
assign: [{{region: all, section: f}}]
regions:
  base: {{box: [-0.01, 0.01, -1, 1, -1, 1]}}
constraints:
  - {{region: all, dofs: [0, 1, 0, 1, 0, 1]}}
  - {{region: base, dofs: [1, 1, 1, 1, 1, 1]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, n_steps: 50}}
output: {{path: out.h5, monitors: {{}}}}
""", encoding="utf-8")
    s = _solver(str(yp))
    res = modal.analyse(s, n_modes=2)
    I = b * h ** 3 / 12.0                          # bending in the local-y (global z) plane
    exact = 1.875104 ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * I / (rho * b * h))
    # 8 / 16 / 32 elements (and fibres across the depth): -1.5 / -0.4 / -0.14 %
    assert res.frequencies[0] == pytest.approx(exact, rel=2e-2)
    assert res.participation[0, 2] > 0.5


def test_irvines_cable_modes_at_the_catenary_state(tmp_path):
    """A 20-element cable of case_22's family relaxed to its catenary, then
    asked for its in-plane modes: the first antisymmetric one is the
    string's 2 pi / L sqrt(H/m), the symmetric ones solve Irvine's
    tan(w/2) = w/2 - (4/lambda^2)(w/2)^3 with
    lambda^2 = (mgL/H)^2 L / (H L_e / EA). Asked at the REFERENCE
    configuration the cable would be a slack straight chord with no
    transverse stiffness at all.

    The cable is 0.3% longer than its span (sag 4%): Irvine's theory is a
    SHALLOW-cable theory, and on case_22's own 8.8% sag the model's first
    antisymmetric mode sits 2.9% below it (0.7% here, the lumped mass's
    share) while the symmetric ones agree to 1% -- the discrete catenary
    is the exact object and the theory is the approximation."""
    from generate_cable_catenary import build
    msh = pathlib.Path(tmp_path) / "cat.msh"
    info = build(str(msh), n=20, L0=100.3)
    yp = pathlib.Path(tmp_path) / "cat.yaml"
    yp.write_text(f"""
name: catenary_20
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  steel: {{type: linear_elastic, E: 2.0e10, nu: 0.3, rho: 7850.0}}
sections:
  main: {{type: bar, material: steel, shape: general, area: 0.05}}
assign:
  - {{region: cable, section: main, unstressed_length: 100.3, tension_only: true}}
regions:
  ends: {{union: [left, right]}}
constraints:
  - {{region: ends, dofs: [1, 1, 1, 1, 1, 1]}}
  - {{region: all, dofs: [0, 1, 0, 1, 1, 1]}}
loads:
  - {{region: all, gravity: [0.0, 0.0, -9.80665]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, max_steps_per_load: 200000,
          stage_ramp_iterations: 50, tolerance: 1.0e-6, kinetic_tolerance: 1.0e-9, n_steps: 100}}
output:
  path: out.h5
  monitors:
    H: {{kind: reaction, region: left, dof: x}}
""", encoding="utf-8")
    s = _solver(str(yp))
    s.run(output_every=1, progress_every=10 ** 9)
    H = -s.monitors["H"].values[-1]
    assert H == pytest.approx(info["H"], rel=5e-3)
    res = modal.analyse(s, n_modes=4)
    L, EA = 100.0, 2.0e10 * 0.05
    m = 7850.0 * 0.05                               # per metre of the (chord) mesh
    # antisymmetric: the string's second mode, lumped on 20 segments (-0.4%)
    f_anti = 2 * np.pi / L * np.sqrt(H / m) / (2 * np.pi)
    # symmetric: Irvine eq. 4.20 with lambda^2
    d = info["sag_mid"]
    L_e = L * (1 + 8 * (d / L) ** 2)
    lam2 = (m * 9.80665 * L / H) ** 2 * L / (H * L_e / EA)
    g = lambda w: np.tan(w / 2) - w / 2 + (4 / lam2) * (w / 2) ** 3
    # the roots of the transcendental equation, skipping tan's poles
    ws = np.linspace(0.1, 4 * np.pi, 4000); vals = np.array([g(w) for w in ws])
    roots = [brentq(g, a, b) for a, b, va, vb in zip(ws[:-1], ws[1:], vals[:-1], vals[1:])
             if va * vb < 0 and abs(va) < 50 and abs(vb) < 50]
    f_sym = [r / L * np.sqrt(H / m) / (2 * np.pi) for r in roots[:2]]
    assert lam2 > 4 * np.pi ** 2                    # past the crossover: antisymmetric first
    assert res.frequencies[0] == pytest.approx(f_anti, rel=1.5e-2)
    assert res.frequencies[1] == pytest.approx(f_sym[0], rel=1e-2)
    assert res.frequencies[2] == pytest.approx(f_sym[1], rel=1.5e-2)    # the second symmetric (-1.1%: 20 lumps)
    assert res.note == ""
    # at the drawing (a slack straight chord) there is no transverse
    # stiffness: the lowest modes are rigid-ish, far below the string's
    ref = modal.analyse(s, n_modes=2, x=np.zeros(s.ndof))
    assert ref.frequencies[0] < 0.2 * f_anti


def test_the_suspension_bridge_vertical_modes_meet_the_deflection_theory(tmp_path):
    """case_23's bridge with ten hangers, at its dead-load state (the
    closed-form cable lengths make the mesh that state), against the
    linearised deflection theory's eigenproblem on a fine grid with the
    same mass per metre (benchmarks/generate_suspension_bridge.vertical_modes).
    The first antisymmetric mode has h = 0 and is the string-plus-beam
    sqrt((EI k^4 + H k^2)/m); the symmetric ones carry the cable's tension
    increment. At the drawing, without the dead load, there would be no
    cable tension and the modes would be those of a slack system."""
    import yaml
    from generate_suspension_bridge import build, vertical_modes
    n = 10
    msh = pathlib.Path(tmp_path) / "bridge.msh"
    info = build(str(msh), n=n)
    doc = yaml.safe_load((ROOT / "cases" / "case_23_suspension_bridge.yaml").read_text(encoding="utf-8"))
    doc["mesh"]["path"] = str(msh)
    doc["output"]["path"] = str(pathlib.Path(tmp_path) / "out.h5")
    doc["loads"][0]["vector"] = [0.0, 0.0, -info["deck_interior_load"] * (n - 1)]
    doc["steps"] = doc["steps"][:1]
    for a in doc["assign"]:
        a.pop("unstressed_lengths", None)
        if a["region"] == "main_cable":
            a["unstressed_lengths"] = [float(v) for v in info["Lu_cable"]]
        if a["region"] == "hangers":
            a["unstressed_lengths"] = [float(v) for v in info["Lu_hangers"]]
    yp = pathlib.Path(tmp_path) / "bridge.yaml"
    yp.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    s = _solver(str(yp))
    s.run(output_every=1, progress_every=10 ** 9)
    res = modal.analyse(s, n_modes=4)
    ref = vertical_modes(4, n=n)
    # ten members: the lumped discretisation is soft by 0.8 / 1.1 / 4 % on
    # the first three (the third is three half-waves on ten lumps; the
    # shipped 30-member case is within 0.8 / 0.3 / 0.7 / 1.1 % on four)
    assert res.frequencies[0] == pytest.approx(ref[0], rel=2e-2), (res.frequencies, ref)
    assert res.frequencies[1] == pytest.approx(ref[1], rel=2e-2)
    assert res.frequencies[2] == pytest.approx(ref[2], rel=5e-2)
    # the first mode is antisymmetric (no net vertical mass), the second symmetric
    assert res.participation[0, 2] < 1e-3 and res.participation[1, 2] > 0.5
    # the cable's interior nodes have free, massless, stiffness-less rotations: dropped
    assert res.n_dropped == n - 1


def test_the_cli_reports_and_writes_the_modes(tmp_path, capsys):
    from vfife.cli import main
    import h5py
    L, E, I, A, rho = 10.0, 2.0e11, 1.0e-4, 0.01, 7850.0
    msh = _line(tmp_path, 10, L)
    yp = pathlib.Path(tmp_path) / "ss.yaml"
    yp.write_text(f"""
name: ss
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  s: {{type: linear_elastic, E: {E}, nu: 0.3, rho: {rho}}}
sections:
  b: {{type: beam, material: s, shape: general, area: {A}, iy: {I}, iz: {I}, j: {2 * I}}}
assign: [{{region: all, section: b}}]
regions:
  left: {{box: [-0.01, 0.01, -1, 1, -1, 1]}}
  right: {{box: [{L - 0.01}, {L + 0.01}, -1, 1, -1, 1]}}
constraints:
  - {{region: all, dofs: [0, 1, 0, 1, 0, 1]}}
  - {{region: left, dofs: [1, 1, 1, 1, 0, 1]}}
  - {{region: right, dofs: [0, 1, 1, 1, 0, 1]}}
loads:
  - {{region: all, gravity: [0, 0, -9.81]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, n_steps: 50, max_steps_per_load: 20000,
          stage_ramp_iterations: 1, tolerance: 1.0e-6, kinetic_tolerance: 1.0e-9}}
output:
  path: {pathlib.Path(tmp_path) / 'ss.h5'}
  fields: [displacement, rotation]
  monitors: {{}}
""", encoding="utf-8")
    out = pathlib.Path(tmp_path) / "modes.h5"
    rc = main(["modal", str(yp), "--modes", "3", "-o", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and out.exists() and "mode    f [Hz]" in text
    with h5py.File(out) as h:
        f = h["frequency"][...]; sh = h["shape"][...]
        assert f.shape == (3,) and sh.shape == (3, 11, 6)
    exact = (np.pi / L) ** 2 * np.sqrt(E * I / (rho * A)) / (2 * np.pi)
    assert f[0] == pytest.approx(exact, rel=1e-2)
    # at a result's state: the beam hangs under gravity, and (linear
    # elastic, small sag) the modes are the same to the finite difference
    from vfife.api import run_case
    run_case(str(yp), progress_every=10 ** 9)
    rc = main(["modal", str(yp), "--modes", "3", "--result", str(pathlib.Path(tmp_path) / "ss.h5"),
               "-o", str(out)])
    assert rc == 0 and "frame" in capsys.readouterr().out
    with h5py.File(out) as h:
        assert h["frequency"][0] == pytest.approx(f[0], rel=1e-3)
