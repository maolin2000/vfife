"""Prestressing tendons in the fibre beam.

A tendon is a bonded steel fibre with a position of its own in every
element -- the profile [[s, y, z], ...] sampled at the element's midpoint
-- and an initial strain, the prestress over the tendon modulus. Nothing
else changes: the section integrates it like a bar, so a draped tendon's
moment P e(x) steps from element to element and those steps are the
balancing load, lumped at the nodes.

The anchors are closed forms on a simply supported beam:
    parabolic tendon, e = 0 at the ends and e_m at midspan:
        camber  = 5 P e_m L^2 / (48 E I)      (w_eq = 8 P e_m / L^2)
    straight tendon at e:  M = P e everywhere, shortening P L / (EA)
and the post-tensioning friction profile the embedded tendons already use.
"""
import pathlib

import numpy as np
import pytest

from vfife.api import _with_element, prepare_case
from vfife.core.structural import make_structural_solver

L, B, H = 10.0, 0.3, 0.6           # span; height along local z, width (depth) along local y
E_C, E_P = 30.0e9, 195.0e9
A_P, SIGMA = 1.0e-3, 1.0e9         # 1 MN
IZ = B * H ** 3 / 12.0             # bending in the local-y (global z) plane


def _line_mesh(td, n):
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


def _case(td, n, tendon, backend="numpy", steps=6000, concrete=None, strand=None):
    """`concrete` / `strand`: the fibre and tendon laws; linear_elastic and
    rebar by default, which have no scalar twin -- the kernel tests use
    elastic steel02 laws in their place."""
    concrete = concrete or f"{{type: linear_elastic, E: {E_C}, nu: 0.0, rho: 2500}}"
    strand = strand or f"{{type: rebar, E: {E_P}, yield_strength: 1.7e9, hardening: 0.0, rho: 7850}}"
    msh = _line_mesh(td, n)
    yp = pathlib.Path(td) / f"t{n}.yaml"
    yp.write_text(f"""
name: tendon_beam
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  c: {concrete}
  p: {strand}
sections:
  strand: {{type: bar, material: p, shape: general, area: {A_P}}}
  girder:
    type: fiber
    material: c
    shape: rect
    width: {H}
    height: {B}
    ny: 24
    nz: 6
    tendons:
      - {tendon}
assign: [{{region: all, section: girder}}]
regions: {{}}
constraints:
  - {{region: all, dofs: [0, 1, 0, 1, 0, 1]}}
  - {{region: left, dofs: [1, 1, 1, 1, 0, 1]}}
  - {{region: right, dofs: [0, 1, 1, 1, 0, 1]}}
solver: {{backend: {backend}, mode: relaxation, load_steps: 1, max_steps_per_load: {steps},
          stage_ramp_iterations: 1, tolerance: 1.0e-6, kinetic_tolerance: 1.0e-10, n_steps: 50}}
output:
  path: out.h5
  monitors:
    w_mid: {{kind: displacement, region: mid, dof: z}}
    u_right: {{kind: displacement, region: right, dof: x}}
""", encoding="utf-8")
    return str(yp)


def _solver(path):
    prep = prepare_case(path)
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


def _run(path):
    s = _solver(path)
    s.run(output_every=1, progress_every=10 ** 9)
    return s


PARABOLA = ("{section: strand, profile: [[0, 0, 0], [5, -0.2, 0], [10, 0, 0]], "
            "interpolation: quadratic, prestress: %g, elastic_shortening: compensate}" % SIGMA)
STRAIGHT = ("{section: strand, profile: [[0, -0.2, 0], [10, -0.2, 0]], "
            "prestress: %g, elastic_shortening: compensate}" % SIGMA)
#: the concrete as an elastic law that HAS a numba twin (linear_elastic and
#: rebar have none): steel02 far below yield is Hooke's law
ELASTIC_TWIN = f"{{type: steel02, E: {E_C}, Fy: 1.0e12, b: 0.0, rho: 2500}}"
STRAND_TWIN = f"{{type: steel02, E: {E_P}, Fy: 1.7e9, b: 0.0, rho: 7850}}"


def test_a_tendon_is_a_steel_row_with_its_own_position_per_element(tmp_path):
    s = _solver(_case(tmp_path, 10, PARABOLA))
    assert s.n_steel == 1 and len(s.tendons) == 1
    col = s.tendons[0]["column"]
    x_mid = 0.5 * (np.arange(10) + np.arange(1, 11))          # element midpoints, m
    e_exp = -0.2 * (1.0 - (2.0 * x_mid / L - 1.0) ** 2)      # the parabola
    assert np.allclose(s.fiber_y[:, col], e_exp, atol=1e-12)
    assert np.allclose(s.fiber_z[:, col], 0.0)
    assert np.allclose(s.fiber_area[:, col], A_P)
    # the initial strain is the prestress over the tendon modulus, plus the
    # compensation for the section's elastic shortening at the tendon
    pre = s.fiber_prestrain[:, col]
    assert np.all(pre > SIGMA / E_P) and np.all(pre < 1.10 * SIGMA / E_P)   # 8% at midspan here
    # every other fibre carries none
    other = np.ones(s.n_fibers, bool); other[col] = False
    assert np.all(s.fiber_prestrain[:, other] == 0.0)


def test_a_straight_tendon_bends_the_concrete_under_a_uniform_P_e(tmp_path):
    """Prestress is self-equilibrated: the SECTION resultants are zero (no
    external load, no reactions), the concrete alone carries -P at e, and
    the beam takes the uniform-moment shape: camber P e L^2 / 8 E I, end
    rotations P e L / 2 E I, and the roller end moves in by P L / EA."""
    s = _run(_case(tmp_path, 8, STRAIGHT))
    _f, actions = s.internal_force()
    P, e = SIGMA * A_P, 0.2
    assert np.all(np.abs(actions[:, 0]) < 1e-4 * P) and np.all(np.abs(actions[:, 5]) < 1e-4 * P * e)
    assert s.monitors["w_mid"].values[-1] == pytest.approx(P * e * L ** 2 / (8.0 * E_C * IZ), rel=2e-3)
    rot = s.dofs.view(s.q)[:, 4]                                 # rotation about global y
    assert abs(rot[0]) == pytest.approx(P * e * L / (2.0 * E_C * IZ), rel=2e-3)
    # the roller end: the concrete's shortening under -P (the fibre grid's
    # area is the gross one) plus the chord shortening of the bent axis,
    # kappa^2 L^3 / 24 for a uniform curvature -- the element is co-rotational
    # and sees it (3% here; 1.852 mm becomes 1.916 mm)
    kappa = P * e / (E_C * IZ)
    u = P * L / (E_C * B * H) + kappa ** 2 * L ** 3 / 24.0
    assert s.monitors["u_right"].values[-1] == pytest.approx(-u, rel=3e-3)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_parabolic_tendon_cambers_the_beam_by_5PeL2_over_48EI(tmp_path, backend):
    """The equivalent load of a parabolic tendon is uniform, 8 P e / L^2
    upward, and the simply supported beam rises 5 w L^4 / 384 EI."""
    kw = {}
    if backend == "numba":
        pytest.importorskip("numba")
        kw = {"concrete": ELASTIC_TWIN, "strand": STRAND_TWIN}
    s = _run(_case(tmp_path, 16, PARABOLA, backend=backend, steps=30000, **kw))
    if backend == "numba":
        assert s._kernel is not None, s.kernel_status
    P, e = SIGMA * A_P, 0.2
    camber = 5.0 * P * e * L ** 2 / (48.0 * E_C * IZ)
    assert s.monitors["w_mid"].values[-1] == pytest.approx(camber, rel=3e-3)


def test_the_polyline_of_eccentricities_converges_with_the_mesh(tmp_path):
    P, e = SIGMA * A_P, 0.2
    camber = 5.0 * P * e * L ** 2 / (48.0 * E_C * IZ)
    err = []
    for n in (4, 8, 16):
        s = _run(_case(tmp_path, n, PARABOLA, steps=30000))
        err.append(abs(s.monitors["w_mid"].values[-1] / camber - 1.0))
    assert err[0] > err[1] > err[2] and err[2] < 5e-3


def test_kernel_and_numpy_paths_agree_with_a_tendon(tmp_path):
    pytest.importorskip("numba")
    s = _solver(_case(tmp_path, 8, PARABOLA, backend="numba", concrete=ELASTIC_TWIN, strand=STRAND_TWIN))
    assert s._kernel is not None, s.kernel_status
    rng = np.random.default_rng(3)
    s.q[:] = 1e-3 * rng.standard_normal(s.ndof)
    fk, ak = s._internal_force_kernel()
    ft, at = s._internal_force_true()
    assert np.allclose(fk, ft, rtol=1e-10, atol=1e-6 * np.abs(ft).max())
    assert np.allclose(ak, at, rtol=1e-10, atol=1e-9 * np.abs(at).max())


def test_jack_and_friction_give_the_post_tensioning_profile(tmp_path):
    """sigma(s) = sigma_con exp(-(mu theta + k s)) from the jacking end,
    on the tendon's own polyline; a straight tendon has theta = 0."""
    tendon = ("{section: strand, profile: [[0, -0.2, 0], [10, -0.2, 0]], "
              "jack: 1.3e9, friction: {mu: 0.2, k: 0.01, jack: start}}")
    s = _solver(_case(tmp_path, 10, tendon))
    sig = s.tendons[0]["sigma"]
    s_mid = 0.5 * (np.arange(10) + np.arange(1, 11))
    assert np.allclose(sig, 1.3e9 * np.exp(-0.01 * s_mid), rtol=1e-12)
    assert np.allclose(s.fiber_prestrain[:, s.tendons[0]["column"]], sig / E_P)
    # jacking from both ends: the smaller loss wins, symmetric about midspan
    tendon = tendon.replace("jack: start", "jack: both")
    s = _solver(_case(tmp_path, 10, tendon))
    sig = s.tendons[0]["sigma"]
    assert np.allclose(sig, sig[::-1]) and sig.min() == pytest.approx(1.3e9 * np.exp(-0.01 * 4.5), rel=1e-12)


def test_a_model_without_tendons_carries_no_initial_strain(tmp_path):
    yp = _case(tmp_path, 4, STRAIGHT)
    text = pathlib.Path(yp).read_text(encoding="utf-8").replace("    tendons:\n      - " + STRAIGHT + "\n", "")
    pathlib.Path(yp).write_text(text, encoding="utf-8")
    s = _solver(yp)
    assert not s.tendons and s.n_steel == 0
    assert np.all(s.fiber_prestrain == 0.0)


def test_the_input_is_checked(tmp_path):
    with pytest.raises(ValueError, match="ONE of"):
        prepare_case(_case(tmp_path, 4, "{section: strand, profile: [[0, 0, 0], [10, 0, 0]]}"))
    with pytest.raises(ValueError, match="increase"):
        prepare_case(_case(tmp_path, 4, "{section: strand, profile: [[0, 0, 0], [0, -0.1, 0]], prestress: 1e9}"))
    with pytest.raises(ValueError, match="three points"):
        prepare_case(_case(tmp_path, 4, "{section: strand, profile: [[0, 0, 0], [10, 0, 0]], "
                                        "interpolation: quadratic, prestress: 1e9}"))
    with pytest.raises(ValueError, match="bar section"):
        prepare_case(_case(tmp_path, 4, "{section: girder, profile: [[0, 0, 0], [10, 0, 0]], prestress: 1e9}"))


# --------------------------------------------------------------------------- #
#  a stress profile along the tendon
# --------------------------------------------------------------------------- #

JACKED = ("{section: strand, profile: [[0, 0, 0], [5, -0.2, 0], [10, 0, 0]], "
          "interpolation: quadratic, jack: 1.3e9, friction: {mu: 0.2, k: 0.01, jack: both}}")


def _table(sig, n):
    s_mid = 0.5 * (np.arange(n) + np.arange(1, n + 1)) * L / n
    return "[" + ", ".join(f"[{a:.12g}, {b:.12g}]" for a, b in zip(s_mid, sig)) + "]"


def test_a_profile_sampled_from_the_friction_solution_reproduces_it(tmp_path):
    """The jack/friction profile, written out at the element midpoints and
    read back as `prestress: [[s, sigma], ...]`, is the same initial strain."""
    n = 10
    ref = _solver(_case(tmp_path, n, JACKED))
    sig = ref.tendons[0]["sigma"]
    tab = JACKED.replace("jack: 1.3e9, friction: {mu: 0.2, k: 0.01, jack: both}",
                         "prestress: " + _table(sig, n))
    s = _solver(_case(tmp_path, n, tab))
    np.testing.assert_allclose(s.tendons[0]["sigma"], sig, rtol=1e-11)
    c0, c1 = ref.tendons[0]["column"], s.tendons[0]["column"]
    np.testing.assert_allclose(s.fiber_prestrain[:, c1], ref.fiber_prestrain[:, c0], rtol=1e-11)


def test_a_tendon_cut_into_pieces_is_the_tendon(tmp_path):
    """WHY THE PROFILE EXISTS. A jacked tendon given as two `extent` pieces
    gets two friction solutions, each jacked at its own ends -- a different
    tendon. The same two pieces carrying the one friction solution as a
    profile are the original tendon: same strains, same camber."""
    n = 10
    ref_path = _case(tmp_path, n, JACKED)
    ref = _run(ref_path)
    sig = ref.tendons[0]["sigma"]
    s_mid = 0.5 * (np.arange(n) + np.arange(1, n + 1)) * L / n
    half = lambda lo, hi: ("{section: strand, profile: [[0, 0, 0], [5, -0.2, 0], [10, 0, 0]], "
                           f"interpolation: quadratic, extent: [{lo}, {hi}], prestress: "
                           + _table(sig, n) + "}")
    d = tmp_path / "split"
    d.mkdir()
    yp = _case(d, n, half(0, 4))
    text = pathlib.Path(yp).read_text(encoding="utf-8").replace(
        "      - " + half(0, 4) + "\n", "      - " + half(0, 4) + "\n      - " + half(4, 10) + "\n")
    pathlib.Path(yp).write_text(text, encoding="utf-8")
    s = _run(yp)
    both = np.zeros(n)
    for t in s.tendons:
        both[t["elements"]] += s.fiber_prestrain[t["elements"], t["column"]]
    np.testing.assert_allclose(both, sig / E_P, rtol=1e-11)
    assert s.monitors["w_mid"].values[-1] == pytest.approx(ref.monitors["w_mid"].values[-1], rel=1e-6)


def test_a_constant_profile_is_a_constant_prestress(tmp_path):
    a = _solver(_case(tmp_path, 4, STRAIGHT))
    b = _solver(_case(tmp_path, 4, STRAIGHT.replace("prestress: %g" % SIGMA,
                                                     "prestress: [[0, %g], [10, %g]]" % (SIGMA, SIGMA))))
    np.testing.assert_array_equal(a.fiber_prestrain, b.fiber_prestrain)


def test_a_profile_that_stops_short_is_refused(tmp_path):
    short = STRAIGHT.replace("prestress: %g" % SIGMA, "prestress: [[0, 1e9], [8, 1e9]]")
    with pytest.raises(ValueError, match="covers s = 0 .. 8"):
        _solver(_case(tmp_path, 4, short))
    with pytest.raises(ValueError, match="must increase"):
        prepare_case(_case(tmp_path, 4, STRAIGHT.replace("prestress: %g" % SIGMA,
                                                         "prestress: [[5, 1e9], [5, 1e9]]")))
