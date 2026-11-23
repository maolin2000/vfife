"""What the Hangzhou Bay girder needed and got: the
`patches` fibre section, a tendon anchored inside the member (`extent`),
a section's own torsion constant, the `fiber` monitor, lumped bars, and
the elastic fibre's modulus -- which this girder was the first model to
read with nu != 0.
"""
import pathlib
import sys

import numpy as np
import pytest

from vfife.api import _with_element, prepare_case
from vfife.core.structural import make_structural_solver
from vfife.elements.fiber_section import patch_fibers

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))


def _solver(path):
    prep = prepare_case(path)
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


# ------------------------------------------------------------ the section
def test_patch_fibers_reproduce_the_patch_areas_and_centroids():
    """A unit square, a trapezoid, and a triangle as a collapsed quad: the
    fibre areas sum to the patch areas and the area-weighted centroids
    converge on the exact ones."""
    y, z, a, cells = patch_fibers([{"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "ny": 2, "nz": 3}])
    assert len(a) == 6 and a.sum() == pytest.approx(1.0) and cells.shape == (6, 4, 2)
    y, z, a, _ = patch_fibers([{"corners": [[0, 0], [2, 0], [1, 1], [0, 1]], "ny": 40, "nz": 40}])
    assert a.sum() == pytest.approx(1.5, rel=1e-12)
    assert (a * y).sum() / a.sum() == pytest.approx(7 / 9, rel=2e-3)      # rect 0.5 x 1 + triangle 4/3 x 0.5
    assert (a * z).sum() / a.sum() == pytest.approx(4 / 9, rel=2e-3)
    y, z, a, _ = patch_fibers([{"corners": [[0, 0], [1, 0], [0, 1], [0, 1]], "ny": 30, "nz": 30}])
    assert a.sum() == pytest.approx(0.5, rel=1e-12)
    assert (a * y).sum() / a.sum() == pytest.approx(1 / 3, rel=5e-3)
    with pytest.raises(ValueError, match="corners"):
        patch_fibers([{"corners": [[0, 0], [1, 0], [1, 1]]}])
    with pytest.raises(ValueError, match="degenerate"):
        patch_fibers([{"corners": [[0, 0], [1, 0], [0, 0], [1, 0]], "ny": 2, "nz": 2}])


def test_the_box_girder_section_has_the_hand_properties():
    """generate_hzb_girder: A 9.73 m2 (a 4 m deep, 15.8 m wide box), the
    centroid 2.48 m above the soffit (top-heavy), I_y 22.4 m4, I_z 128."""
    from generate_hzb_girder import section_properties, torsion_constant
    p = section_properties()
    assert p["A"] == pytest.approx(9.734, rel=1e-3)
    assert p["v_c"] == pytest.approx(2.478, rel=1e-3)
    assert p["I_y"] == pytest.approx(22.45, rel=2e-3)
    assert p["I_z"] == pytest.approx(128.3, rel=2e-3)
    assert 25.0 < torsion_constant() < 35.0


# ------------------------------------------------------- the model checks
CASE = pathlib.Path(__file__).resolve().parents[1] / "cases" / "case_24b_hzb_girder_modal.yaml"


def test_the_elastic_fibre_answers_with_E_not_the_constrained_modulus(tmp_path):
    """A fibre is in uniaxial STRESS. Handing the 3-D elastic law
    [eps, 0, 0, ...] returned C11 eps -- E (1 - nu) / ((1 + nu)(1 - 2 nu)),
    7% stiff at nu = 1/6 and 35% at 0.3 -- and every fibre test had nu = 0.
    The girder's midspan deflection under 1 MN is P L^3 / 48 E I to the
    fibre grid's own I."""
    s = _solver(str(CASE))
    s.run(output_every=1, progress_every=10 ** 9)
    I = float(s.Iz[0])                      # sum a y^2 of the fibre grid: bending in the vertical plane
    assert I == pytest.approx(21.87, rel=1e-2)
    w = 1.0e6 * 70.0 ** 3 / (48.0 * 3.5e10 * I)
    assert -s.monitors["w_mid"].values[-1] == pytest.approx(w, rel=2e-3)
    assert float(s.J[0]) == pytest.approx(29.51)


def test_the_first_vertical_mode_is_the_girder_s(tmp_path):
    """1.77 Hz with the fibre I against Euler's 1.80 (the lumped rotary
    inertia), Xu's ANSYS 1.85 for the thickened girder; the third mode is
    the second vertical near 4x, the fourth the torsional one at Bredt's
    J -- 150 m4 of polar moment would put it at 17 Hz."""
    from vfife import modal
    s = _solver(str(CASE))
    res = modal.analyse(s, n_modes=4)
    f = res.frequencies
    assert f[0] == pytest.approx(1.80, rel=2.5e-2) and res.participation[0, 2] > 0.8
    assert res.participation[1, 1] > 0.8                    # lateral
    assert 3.6 < f[2] / f[0] < 4.0                          # second vertical, rotary inertia
    assert f[3] == pytest.approx(7.77, rel=3e-2)            # torsion: (1/2L) sqrt(G J / rho I_p)


# ------------------------------------------------- tendons with an extent
L, B, H = 10.0, 0.3, 0.6
E_C, E_P, A_P, SIGMA = 30.0e9, 195.0e9, 1.0e-3, 1.0e9


def _mesh(td, n):
    msh = pathlib.Path(td) / "beam.msh"
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "4",
           '0 1 "left"', '0 2 "right"', '0 3 "mid"', '1 4 "beam"', "$EndPhysicalNames",
           "$Nodes", str(n + 1)]
    out += [f"{k + 1} {L * k / n:.12g} 0 0" for k in range(n + 1)]
    out += ["$EndNodes", "$Elements", str(n + 3), "1 15 2 1 1 1", f"2 15 2 2 2 {n + 1}",
            f"3 15 2 3 3 {n // 2 + 1}"]
    out += [f"{k + 4} 1 2 4 4 {k + 1} {k + 2}" for k in range(n)]
    out += ["$EndElements", ""]
    msh.write_text("\n".join(out), encoding="utf-8")
    return msh


def _case(td, tendon, monitors="", bars=""):
    msh = _mesh(td, 10)
    yp = pathlib.Path(td) / "t.yaml"
    yp.write_text(f"""
name: tendon_extent
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  c: {{type: linear_elastic, E: {E_C}, nu: 0.2, rho: 2500}}
  p: {{type: rebar, E: {E_P}, yield_strength: 1.7e9, hardening: 0.0, rho: 7850}}
sections:
  strand: {{type: bar, material: p, shape: general, area: {A_P}}}
  girder:
    type: fiber
    material: c
    shape: rect
    width: {H}
    height: {B}
    ny: 24
    nz: 6{bars}
    tendons:
      - {tendon}
assign: [{{region: all, section: girder}}]
regions: {{}}
constraints:
  - {{region: all, dofs: [0, 1, 0, 1, 0, 1]}}
  - {{region: left, dofs: [1, 1, 1, 1, 0, 1]}}
  - {{region: right, dofs: [0, 1, 1, 1, 0, 1]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, max_steps_per_load: 6000,
          stage_ramp_iterations: 1, tolerance: 1.0e-6, kinetic_tolerance: 1.0e-10, n_steps: 50}}
output:
  path: out.h5
  monitors:
    w_mid: {{kind: displacement, region: mid, dof: z}}{monitors}
""", encoding="utf-8")
    return str(yp)


def test_a_bounded_tendon_exists_only_between_its_anchors(tmp_path):
    """A straight tendon over the middle 4 m only: zero area in the other
    six elements, and the camber of a uniform moment M = P e over
    [a, L - a]: the conjugate beam gives w = M (L/2 - a)(L/4 + a/2) / E I."""
    s = _solver(_case(tmp_path, "{section: strand, profile: [[3, -0.2, 0], [7, -0.2, 0]], extent: [3, 7], prestress: %g}" % SIGMA))
    col = s.tendons[0]["column"]
    assert np.count_nonzero(s.fiber_area[:, col]) == 4
    assert np.all(s.fiber_area[3:7, col] == A_P) and np.all(s.fiber_prestrain[[0, 1, 2, 7, 8, 9], col] == 0.0)
    s.run(output_every=1, progress_every=10 ** 9)
    I = B * H ** 3 / 12.0
    P_eff = SIGMA * A_P * float(np.mean(s.fiber_stress_rows()[3:7, :, col])) / SIGMA   # after elastic shortening
    a = 3.0
    w = P_eff * 0.2 * (L / 2 - a) * (L / 4 + a / 2) / (E_C * I)
    assert s.monitors["w_mid"].values[-1] == pytest.approx(w, rel=1e-2)
    with pytest.raises(ValueError, match="extent"):
        _solver(_case(tmp_path, "{section: strand, profile: [[3, -0.2, 0], [7, -0.2, 0]], extent: [3.1, 3.2], prestress: 1.0e9}"))


def test_the_fiber_monitor_reads_the_plane_section_at_a_point(tmp_path):
    """Under a uniform moment the strain at y is -y kappa with kappa = M / E I,
    and the stress there is E times it; the monitor reads the station AT
    the node (lobatto) and agrees with the fibre stress rows."""
    mon = """
    eps_top: {kind: fiber, region: mid, field: strain, at: [0.3, 0.0]}
    eps_bot: {kind: fiber, region: mid, field: strain, at: [-0.3, 0.0]}
    sig_bot: {kind: fiber, region: mid, field: stress, at: [-0.3, 0.0]}"""
    s = _solver(_case(tmp_path, "{section: strand, profile: [[0, -0.2, 0], [10, -0.2, 0]], prestress: %g}" % SIGMA, monitors=mon))
    s.run(output_every=1, progress_every=10 ** 9)
    top, bot, sig = (s.monitors[k].values[-1] for k in ("eps_top", "eps_bot", "sig_bot"))
    # the concrete carries -P at e = -0.2: eps = -P/EA - P e y / EI (the tendon row excluded)
    P = float(np.mean(s.fiber_stress_rows()[:, :, s.tendons[0]["column"]])) * A_P
    A, I = B * H, B * H ** 3 / 12.0
    assert bot == pytest.approx(-P / (E_C * A) - P * 0.2 * 0.3 / (E_C * I), rel=2e-2)
    assert top == pytest.approx(-P / (E_C * A) + P * 0.2 * 0.3 / (E_C * I), rel=2e-2)
    assert sig / E_C == pytest.approx(bot, rel=6e-2)        # the nearest fibre sits 12.5 mm in from the face


def test_a_bar_count_lumps_identical_bars(tmp_path):
    bars = "\n    bars: [{y: 0.25, z: 0.0, section: strand, count: 3}, {y: -0.25, z: 0.0, section: strand}]"
    s = _solver(_case(tmp_path, "{section: strand, profile: [[0, -0.2, 0], [10, -0.2, 0]], prestress: 1.0e9}", bars=bars))
    lay = s._layouts[0]
    areas = sorted(float(s.fiber_area[0, lay.n_c + j]) for j in range(lay.n_s))
    assert areas == pytest.approx([A_P, A_P, 3 * A_P])


def test_the_torsion_constant_sets_the_torsional_stiffness(tmp_path):
    from vfife.api import prepare_case as pc
    path = _case(tmp_path, "{section: strand, profile: [[0, -0.2, 0], [10, -0.2, 0]], prestress: 1.0e9}")
    s = _solver(path)
    assert float(s.J[0]) == pytest.approx(float(s.Iy[0] + s.Iz[0]))
    txt = pathlib.Path(path).read_text(encoding="utf-8").replace("    nz: 6", "    nz: 6\n    torsion_constant: 0.001")
    pathlib.Path(path).write_text(txt, encoding="utf-8")
    s2 = _solver(path)
    assert float(s2.J[0]) == pytest.approx(0.001)
    # the reference stiffness's torsional pair is G J / L
    k = s2._reference_k(0)
    assert k[3, 3] == pytest.approx(E_C / (2 * 1.2) * 0.001 / 1.0)


def test_a_draped_tendon_is_drawn_as_the_drape_not_a_staircase(tmp_path):
    """The sweep's bars series. The section integration
    samples a tendon once per element, at the midpoint; drawing that value
    at BOTH ends of the element makes every element a horizontal chord and
    the drape a staircase -- which is why the tendons of the Hangzhou Bay
    girder came out of ParaView looking straight. The drawn polyline now
    runs between the profile's values at the element's two ends, so it is
    continuous at the nodes and lands on the profile."""
    from vfife.sweep import Sweep
    path = _case(tmp_path, "{section: strand, profile: [[0, 0, 0], [5, -0.2, 0], [10, 0, 0]], "
                           "interpolation: quadratic, prestress: 1.0e9}")
    sw = Sweep(path)
    pts, lines, owner, cols, stat = sw.bars_frame(np.zeros((11, 3)), np.zeros((11, 3)))
    col = sw.sol.tendons[0]["column"]
    seg = lines[cols == col]
    assert len(seg) >= 10
    ends = np.unique(seg)
    x, z = pts[ends, 0], pts[ends, 2]      # member along x, local y -> global z
    para = -0.2 * (1.0 - (2.0 * x / L - 1.0) ** 2)
    node = np.abs(x - np.round(x)) < 1e-9                  # the element boundaries
    assert np.allclose(z[node], para[node], atol=1e-9)     # ON the profile there
    assert np.all(np.abs(z - para) < 3e-3)                 # the stations between: the chord
    assert abs(z[np.argmin(x)]) < 1e-12 and abs(z[np.argmax(x)]) < 1e-12   # at the anchors, not 0.036 in
    # continuity: every interior node carries ONE drawn point, not two
    for xi in (1.0, 2.0, 5.0, 9.0):
        here = np.abs(x - xi) < 1e-9
        assert here.sum() >= 1 and np.ptp(z[here]) < 1e-12
    # a straight bar is still drawn at its own constant position
    bars = "\n    bars: [{y: 0.1, z: 0.0, section: strand}]"
    sw2 = Sweep(_case(tmp_path, "{section: strand, profile: [[0, -0.2, 0], [10, -0.2, 0]], prestress: 1.0e9}", bars=bars))
    p2, l2, o2, c2, s2 = sw2.bars_frame(np.zeros((11, 3)), np.zeros((11, 3)))
    assert set(np.round(p2[np.unique(l2[c2 == c2.min()]), 2], 6)) == {0.1}


def test_the_generator_reproduces_every_committed_mesh_asset(tmp_path):
    """Byte-for-byte, all four of them.

    THIS IS WHY. `hzb_girder_3span_14_piers.msh` and `..._bearings.msh` are
    written by the same `main()` from the same `--n-pier`, and the committed
    piers mesh had FOUR pier elements while the committed bearings mesh had
    six -- so no single run of the generator could produce both, and merely
    running it silently rewrote case_24d's mesh from 4 elements to 6 and its
    answers with it. That is how a container that had run the generator ended
    up building case_24d's pier regions against a mesh the repository does not
    contain, and the case then failed with "region covers no complete Line2
    element" on a machine that had the committed one.

    A generated asset that its generator does not reproduce is a fork in the
    data with nothing watching it. The piers mesh is now six elements like the
    bearings mesh, and this test is what keeps them in step.
    """
    import sys
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import generate_hzb_girder as hz

    hz.main(["--out", str(tmp_path)])
    committed = sorted((ROOT / "benchmarks" / "assets").glob("hzb_girder_*.msh"))
    assert len(committed) == 4
    for path in committed:
        fresh = tmp_path / path.name
        assert fresh.exists(), f"{path.name} is committed but the generator does not write it"
        assert fresh.read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), (
            f"{path.name} differs from what the generator produces; regenerate it "
            "or give the asset its own parameters")
