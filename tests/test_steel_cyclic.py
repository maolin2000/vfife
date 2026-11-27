"""The cyclic steel law of Cai et al. 2023 §3.

Four curves, two reloading branches, five hysteresis criteria and a fatigue
model. What is asserted here is each piece against the equation it comes from,
and -- for the three equations that do not close as printed -- the correction
this implementation makes, with the number that shows why it was needed.

The material invariants (tests/test_material_invariants.py) ask the questions
that apply to every law; this file asks the ones that only apply to this one.
"""

import numpy as np
import pytest

from vfife.materials import steel_cyclic as sc
from vfife.materials.base import create
from vfife.base.compat import trapezoid


def law(**kw):
    return create("steel_cyclic", **{"grade": "Q345B", "E": 2.06e11, **kw})


def drive(m, path, n=600):
    """Walk a strain path; return (eps, sigma, branch, acc_c, failed) rows."""
    st = m.init_state(1)
    eps = np.zeros((1, 6))
    rows = []
    prev = 0.0
    for target in path:
        for e in np.linspace(prev, target, n)[1:]:
            eps[0, 0] = e
            s = float(m.update(eps, st, 1.0)[0, 0])
            rows.append((e, s, float(st[0, 2]), float(st[0, 7]), float(st[0, 8])))
        prev = target
    return np.array(rows), st


# --------------------------------------------------------------------------- #
#  the monotonic curve, eq. (10)
# --------------------------------------------------------------------------- #

def test_the_monotonic_curve_has_its_four_stages_and_no_step_between_them():
    m = law()
    ey, fy, fu = m.eps_y, m.fy, m.fu
    assert m.monotonic(0.5 * ey) == pytest.approx(0.5 * fy)
    assert m.monotonic(ey) == pytest.approx(fy)
    assert m.monotonic(m.k1 * ey) == pytest.approx(fy)          # end of the plateau
    assert m.monotonic(m.k2 * ey) == pytest.approx(fu)          # end of hardening
    assert m.monotonic(2.0 * m.k2 * ey) == pytest.approx(fu)
    e = np.linspace(1.0e-6, 1.2 * m.k2 * ey, 20001)
    s = np.array([m.monotonic(x) for x in e])
    assert np.all(np.diff(s) >= -1e-6), "the monotonic curve must not fall"
    # Lipschitz at E, which is the only bound that means anything here: the
    # elastic stage rises by E de per sample, so a fixed stress bound would
    # either accuse the elastic branch or be too loose to catch a real step.
    assert np.abs(np.diff(s)).max() <= m.E * np.diff(e)[0] * (1.0 + 1e-9)


def test_the_printed_hardening_coefficient_would_drop_the_curve_by_a_gigapascal():
    """Eq. (10) is printed with `E(1-k1)/(eps_y (k2-k1)^2)`. That coefficient
    puts the hardening branch at f_u + f_y(1 - k1) where the plateau leaves off
    at f_y -- for Q345B a fall of 1.5 GPa at first yield, which is 3.5 times the
    yield stress and the wrong way. The number is pinned here so the choice
    made in materials/steel_cyclic.py is a decision on the record."""
    m = law()
    printed = m.fu + m.fy * (1.0 - m.k1)
    assert printed == pytest.approx(-887.5e6, rel=1e-3)
    assert m.monotonic(m.k1 * m.eps_y) == pytest.approx(m.fy)
    # and the coefficient used here IS the printed one with k3 for k1, whenever
    # f_u = k3 f_y -- which is the curve their Fig. 4 draws
    n = create("steel_cyclic", E=2.06e11, rho=7850.0, fy=421.0e6, k3=1.35,
               k1=4.5, k2=45.0, a=5.796, b0=2.486, b1=0.608, b2=-0.00046,
               eps_cri=0.36, lambda_dsps=0.28)
    assert n.fu == pytest.approx(1.35 * n.fy)
    e = 0.5 * (n.k1 + n.k2) * n.eps_y
    printed_with_k3 = n.fu + n.E * (1.0 - n.k3) / (n.eps_y * (n.k2 - n.k1) ** 2) \
        * (e - n.k2 * n.eps_y) ** 2
    assert n.monotonic(e) == pytest.approx(printed_with_k3, rel=1e-12)


# --------------------------------------------------------------------------- #
#  the hysteresis skeleton, eq. (11)
# --------------------------------------------------------------------------- #

def test_the_published_skeleton_does_not_meet_the_elastic_line():
    """3% of f_y, at first yield, on every reloading path -- the shape v0.1
    showed can trap an explicit solver. `skeleton_fit='shift'` removes it."""
    raw = law(skeleton_fit="raw")
    step = raw.skeleton(raw.eps_y * (1.0 + 1e-9)) / raw.fy - 1.0
    assert step == pytest.approx(0.0302, abs=5e-4)
    fixed = law()
    assert fixed.skeleton(fixed.eps_y * (1.0 + 1e-9)) == pytest.approx(fixed.fy, rel=1e-6)
    # the shift is a constant: the slope is the paper's everywhere
    for e in (0.005, 0.01, 0.02):
        assert (raw.skeleton(e) - fixed.skeleton(e)) == pytest.approx(fixed.shift * fixed.fy)


def test_the_step_shows_up_on_a_path_whose_peak_sits_at_yield():
    """And only on such a path, which is why the v0.1 sweep passes either
    way: below the peak the reloading curve covers the strain range, so the
    skeleton is never evaluated near eps_y unless the peak itself is there.
    Drive one that is -- reverse just under yield, then push past it -- and the
    raw form steps 13 MPa in one sample, 35 times the elastic slope. That is
    the `bond_ceb` shape of v0.1, at first yield, on every such reversal."""
    def worst(fit):
        m = law(skeleton_fit=fit)
        st = m.init_state(1)
        eps = np.zeros((1, 6))
        rows, prev = [], 0.0
        for t in (0.95 * m.eps_y, -0.5 * m.eps_y, 3.0 * m.eps_y):
            for e in np.linspace(prev, t, 4000)[1:]:
                eps[0, 0] = e
                rows.append((e, float(m.update(eps, st, 1.0)[0, 0])))
            prev = t
        r = np.array(rows)
        d = np.abs(np.diff(r[:, 1]))
        de = np.abs(np.diff(r[:, 0]))
        return float((d / (m.E * de)).max()), float(d.max())

    raw_slope, raw_jump = worst("raw")
    fix_slope, fix_jump = worst("shift")
    # SINCE v1.0 THE RAW FORM NO LONGER STEPS HERE EITHER. The step was the
    # reload line (capped at E, because the secant to a peak at 1.03 f_y over
    # eps_y is steeper than E) arriving at the peak strain under the skeleton
    # and jumping onto it: 13 MPa in one sample, 35 times the elastic slope.
    # The reload now stays on its line at E until it meets the curve, which
    # bridges the paper's 3% for every law, not only the shifted one. The
    # discontinuity of eq. (11) itself is still there, and the test above
    # still measures it.
    assert raw_slope <= 1.0 + 1e-9 and raw_jump <= fix_jump * (1 + 1e-6)
    assert fix_slope <= 1.0 + 1e-9, "the shifted law is Lipschitz at E"


def test_the_cyclic_hardening_is_about_a_fifth_above_the_monotonic_curve():
    """The paper's own reading of the experiment: 'the cyclic hardening stress
    of the experimental specimen is about 20% higher than the stress of
    monotonic loading at the same strain'."""
    m = law()
    r = m.skeleton(10 * m.eps_y) / m.monotonic(10 * m.eps_y)
    assert 1.15 < r < 1.25, r


# --------------------------------------------------------------------------- #
#  unloading and reloading, eqs. (14)-(18)
# --------------------------------------------------------------------------- #

def test_unloading_is_elastic_at_the_initial_modulus():
    """Criterion (2), and it is the one thing in the law with no parameters."""
    m = law()
    st = m.init_state(1)
    eps = np.zeros((1, 6))
    for e in np.linspace(0.0, 0.02, 400)[1:]:
        eps[0, 0] = e
        s_top = float(m.update(eps, st, 1.0)[0, 0])
    e_top = 0.02
    slopes = []
    for e in np.linspace(e_top, e_top - 1.0e-3, 50)[1:]:
        eps[0, 0] = e
        s = float(m.update(eps, st, 1.0)[0, 0])
        slopes.append((s - s_top) / (e - e_top))
        s_top, e_top = s, e
    assert np.allclose(slopes, m.E, rtol=1e-9)


@pytest.mark.parametrize("amp,expect", [(0.004, "legeron"), (0.03, "shi_wang")])
def test_the_reloading_curve_ends_exactly_on_the_peak_point(amp, expect):
    """Criteria (3) and (4): the reloading curve is aimed at the peak point,
    and both of the paper's two curves have to arrive there. Which one runs is
    decided by the secant stiffness E_k of eq. (14), so the loop width picks
    the branch -- that is what the parametrisation is for."""
    m = law()
    rows, st = drive(m, [amp, -amp])
    target = m.skeleton(-amp)
    assert rows[-1, 1] == pytest.approx(target, rel=1e-9)
    e0, s0 = float(st[0, 5]), float(st[0, 6])
    Ek = abs(target - s0) / abs(-amp - e0)
    if expect == "legeron":
        assert m.E / 5.0 <= Ek < m.E
    else:
        assert 0.0 < Ek < m.E / 5.0


def test_the_reloading_curve_stays_between_its_two_ends():
    """No overshoot and no crossing: monotone in the stress from the unloaded
    point to the peak. Fig. 12(b) of the paper is a picture of what happens
    when a reloading curve does cross -- 'excessive softening' and 'curve
    crossing' are the two labels on the model it is compared against."""
    m = law()
    for amp in (0.004, 0.01, 0.03):
        rows, _ = drive(m, [amp, -amp], n=1500)
        turn = int(np.argmax(rows[:, 0]))
        back = rows[turn:]
        reload_leg = back[back[:, 2] == sc.RELOAD]
        assert len(reload_leg) > 50
        assert np.all(np.diff(reload_leg[:, 1]) <= 1e-6), "reloading is not monotone"
        assert reload_leg[:, 1].min() >= m.skeleton(-amp) - 1e-6


def test_eta_is_one_at_the_peak_and_below_one_at_the_start():
    """Eq. (17), the unified function that replaces the two-segment one."""
    for zeta in (0.01, 0.1, 1.0, 10.0):
        assert sc._eta(1.0, zeta) == pytest.approx(1.0)
        assert 0.0 < sc._eta(0.0, zeta) < 1.0
        r = np.linspace(0.0, 1.0, 200)
        assert np.all(np.diff([sc._eta(x, zeta) for x in r]) > 0.0)


# --------------------------------------------------------------------------- #
#  degradation and fracture, eqs. (12), (13)
# --------------------------------------------------------------------------- #

def test_the_degradation_curve_runs_from_fu_at_necking_to_fuf_at_fracture():
    """Eq. (12) as printed gives 2f_u - f_uf at the necking strain and f_u at
    fracture -- both ends wrong. The form used here matches their Fig. 5:
    flat at necking, steep at fracture."""
    m = law()
    eu, euf, fu, fuf = m.eps_u_skel, m.eps_cri, m.fu, m.f_uf
    printed = fu + (fu - fuf) * ((eu - euf) / (eu - euf)) ** 2
    assert printed == pytest.approx(2 * fu - fuf)              # the defect, pinned
    assert sc._degradation(eu, fu, fuf, eu, euf) == pytest.approx(fu)
    assert sc._degradation(euf, fu, fuf, eu, euf) == pytest.approx(fuf)
    mid = np.linspace(eu, euf, 500)
    s = np.array([sc._degradation(x, fu, fuf, eu, euf) for x in mid])
    assert np.all(np.diff(s) < 0.0)
    assert sc._degradation_slope(eu, fu, fuf, eu, euf) == pytest.approx(0.0)


def test_cycling_brings_the_necking_strain_forward():
    """Fig. 5's whole point: eps_u under cyclic loading is where the SKELETON
    reaches f_u, and that is earlier than where the monotonic curve does."""
    m = law()
    assert m.eps_u_skel < m.eps_u_mono
    assert m.eps_u_skel == pytest.approx(0.0361, abs=5e-4)
    assert m.eps_u_mono == pytest.approx(m.k2 * m.eps_y)
    assert m.skeleton(m.eps_u_skel) == pytest.approx(m.fu, rel=1e-6)


def test_a_monotonic_pull_peaks_at_fu_and_fractures_at_the_coupon_strain():
    m = law()
    rows, _ = drive(m, [0.40], n=4000)
    assert rows[:, 1].max() == pytest.approx(m.fu, rel=1e-3)
    at_peak = rows[int(np.argmax(rows[:, 1])), 0]
    assert at_peak == pytest.approx(m.eps_u_mono, rel=0.02)
    broke = rows[rows[:, 4] > 0.5]
    assert len(broke) and broke[0, 0] == pytest.approx(m.eps_cri, rel=0.01)


def test_compressive_cycling_shortens_the_fracture_strain():
    """Eq. (13), the DSPS model, and the reason this law belongs in a collapse
    analysis: the fibre fractures in tension at a strain the coupon never
    saw, because of the compression it went through first."""
    m = law()
    assert m.fracture_strain(0.0) == pytest.approx(m.eps_cri)
    assert m.fracture_strain(1.0) == pytest.approx(np.exp(-m.lambda_dsps) * m.eps_cri)

    plain, _ = drive(m, [0.40], n=3000)
    cycled, st = drive(m, [-0.05, 0.0] * 6 + [0.40], n=800)
    e_plain = plain[plain[:, 4] > 0.5][0, 0]
    e_cycled = cycled[cycled[:, 4] > 0.5][0, 0]
    acc = float(st[0, 7])
    # six excursions to -5%, of which the plastic part is what counts
    assert acc == pytest.approx(0.269, rel=0.05), acc
    assert e_cycled < 0.94 * e_plain
    assert e_cycled == pytest.approx(m.fracture_strain(acc), rel=0.02)


def test_the_stress_is_zero_after_fracture_and_the_latch_says_so():
    """The v0.1 contract: `failure` reports the event the law itself acts
    on, so the element counts fibres instead of guessing from a zero stress."""
    m = law(eps_cri=0.02)
    st = m.init_state(1)
    eps = np.zeros((1, 6))
    seen = False
    for e in list(np.linspace(0, 0.03, 400)) + list(np.linspace(0.03, 0.0, 200)):
        eps[0, 0] = e
        s = float(m.update(eps, st, 1.0)[0, 0])
        failed = bool(m.failure(st)[0] > 0.5)
        if failed:
            assert s == 0.0
            seen = True
        else:
            assert not seen, "the latch let go"
    assert seen


def test_the_fibre_falls_to_a_residual_before_it_is_cut():
    """Why this law was written: a bare rupture latch drops the
    whole stress in one step. Here the degradation curve has already taken it
    down to f_uf, so the step the latch finally makes is half of f_u rather
    than all of it."""
    m = law()
    rows, _ = drive(m, [0.40], n=6000)
    broke = int(np.flatnonzero(rows[:, 4] > 0.5)[0])
    assert rows[broke - 1, 1] == pytest.approx(m.f_uf, rel=0.02)
    assert rows[broke - 1, 1] / m.fu == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------------------- #
#  the hysteresis criteria as a whole
# --------------------------------------------------------------------------- #

def test_the_peak_point_is_isotropic():
    """Criterion (3) with the paper's reading of steel's isotropic hardening:
    a tensile excursion to eps_1 sets the target for the COMPRESSIVE reload at
    -eps_1, which is where Fig. 7 puts point E."""
    m = law()
    rows, st = drive(m, [0.012, -0.012])
    assert float(st[0, 4]) == pytest.approx(0.012, rel=1e-6)
    assert rows[-1, 1] == pytest.approx(m.skeleton(-0.012), rel=1e-9)


def test_a_small_excursion_inside_a_reloading_branch_returns_to_it():
    """Fig. 7's C-D-C: unloading part way through a reload is elastic, and
    coming back resumes the SAME reloading curve rather than starting a new
    one from wherever the excursion ended."""
    m = law()
    straight, _ = drive(m, [0.02, -0.02], n=1200)
    wiggled, _ = drive(m, [0.02, -0.005, -0.002, -0.02], n=1200)
    assert wiggled[-1, 1] == pytest.approx(straight[-1, 1], rel=1e-9)
    # and the excursion itself was elastic
    leg = wiggled[(wiggled[:, 0] > -0.005) & (wiggled[:, 0] < -0.002)]
    assert (leg[:, 2] == sc.ELASTIC).any()


def test_the_four_branches_all_run_in_one_history():
    m = law()
    rows, _ = drive(m, [0.02, -0.005, -0.02, 0.01, 0.05])
    assert set(np.unique(rows[:, 2])) == {sc.VIRGIN, sc.ELASTIC, sc.RELOAD, sc.SKELETON}


def test_repeated_cycles_at_one_amplitude_settle_to_the_same_loop():
    """A stable loop, to machine precision, from the second cycle on.

    The peak point is isotropic and the amplitude is fixed, so nothing in the
    law changes between cycle 2 and cycle 3 except the accumulated compressive
    plastic strain -- which only moves the fracture strain, not the shape. A
    loop that keeps growing or shrinking here would mean a state variable is
    leaking between cycles, which is the defect this shape of test exists to
    find (it is what v0.1 measured as a 16x error in cyclic energy).
    """
    m = law()
    rows, _ = drive(m, [0.01, -0.01] * 4, n=1500)
    n = len(rows) // 8                                # one leg of the path
    # legs 3+4, 5+6, 7+8: each a closed loop from -0.01 back to -0.01. Legs 1
    # and 2 are the virgin excursion and are a different thing.
    cycles = [rows[2 * n:4 * n], rows[4 * n:6 * n], rows[6 * n:8 * n]]
    areas = [abs(float(trapezoid(c[:, 1], c[:, 0]))) for c in cycles]
    assert areas[0] > 0.0
    assert areas[1] == pytest.approx(areas[0], rel=1e-9)
    assert areas[2] == pytest.approx(areas[0], rel=1e-9)
    assert rows[:, 1].max() <= m.skeleton(0.01) + 1.0


# --------------------------------------------------------------------------- #
#  the paper's table
# --------------------------------------------------------------------------- #

def test_the_grades_are_table_1():
    assert set(sc.GRADES) == {"Q345B", "Q235B", "HPS485", "HT440"}
    q = sc.GRADES["Q345B"]
    assert (q["k1"], q["k2"], q["k3"]) == (4.5, 45.0, 1.35)
    assert (q["fy"], q["fu"]) == (421.0e6, 586.0e6)
    assert (q["eps_cri"], q["lambda_dsps"]) == (0.36, 0.28)
    for name, g in sc.GRADES.items():
        assert (g["a"], g["b0"], g["b1"], g["b2"]) == (5.796, 2.486, 0.608, -0.00046), name
        if name == "HT440":
            continue                      # see the next test
        m = law(grade=name)
        assert m.fy < m.fu
        assert m.eps_u_skel < m.eps_u_mono < m.eps_cri


def test_the_one_table_row_whose_numbers_do_not_close_is_refused():
    """HT440: k3 = 1.2 with a measured f_u of 1.60 f_y, and the same four
    Ramberg-Osgood parameters as every other row. With that f_u the skeleton
    needs 12.3% strain to reach it while the monotonic curve is there at
    11.2%, so CYCLING WOULD DELAY NECKING -- the opposite of the Fig. 5 the
    whole degradation model rests on. Refused at construction, with the two
    ways out named, rather than run quietly."""
    with pytest.raises(ValueError, match="DELAY necking"):
        law(grade="HT440")
    # AND RAISING k3 TO MATCH THE MEASURED f_u DOES NOT CLOSE IT EITHER: the
    # Ramberg-Osgood row is the same for all four grades, and that curve
    # cannot reach 800 MPa before 14.8% however k3 is labelled. The parameter
    # that has to move is f_u -- or a, b0, b1, b2, which would mean fitting
    # this steel rather than reusing the table's defaults.
    g = sc.GRADES["HT440"]
    with pytest.raises(ValueError, match="DELAY necking"):
        law(grade="HT440", k3=g["fu"] / g["fy"])
    ok = law(grade="HT440", fu=g["k3"] * g["fy"])
    assert ok.eps_u_skel < ok.eps_u_mono
    assert ok.fu == pytest.approx(601.2e6)


def test_an_unknown_grade_is_refused_with_the_list():
    with pytest.raises(ValueError, match="Q345B"):
        create("steel_cyclic", grade="A36")


def test_the_law_can_be_given_its_eleven_numbers_instead_of_a_grade():
    m = create("steel_cyclic", E=2.0e11, rho=7850.0, fy=3.0e8, fu=4.5e8,
               k1=4.0, k2=40.0, a=5.796, b0=2.486, b1=0.608, b2=-0.00046,
               eps_cri=0.3, lambda_dsps=0.25)
    assert m.monotonic(m.k2 * m.eps_y) == pytest.approx(4.5e8)
    with pytest.raises(ValueError, match="eps_cri"):
        create("steel_cyclic", E=2.0e11, rho=7850.0, fy=3.0e8, fu=4.5e8,
               k1=4.0, k2=40.0, a=5.796, b0=2.486, b1=0.608, b2=-0.00046,
               lambda_dsps=0.25)


# --------------------------------------------------------------------------- #
#  on a fibre section: the v0.1 pipeline, driven by this law
# --------------------------------------------------------------------------- #

def _fibre_bar(td, length=0.2, threshold=1.0, eps_cri=0.02, name="sc"):
    import pathlib

    msh = pathlib.Path(td) / f"{name}.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n2\n"
                   f"1 0 0 0\n2 {length} 0 0\n$EndNodes\n$Elements\n1\n"
                   "1 1 2 1 1 1 2\n$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(td) / f"{name}.yaml"
    yp.write_text(f"""
name: {name}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
materials:
  steel: {{type: steel_cyclic, grade: Q345B, E: 2.06e11, rho: 7850.0,
           eps_cri: {eps_cri}}}
sections:
  s: {{type: fiber, material: steel, shape: rect, width: 0.05, height: 0.05,
      ny: 3, nz: 5, fracture: {{threshold: {threshold}}}}}
assign: [{{region: all, section: s}}]
regions: {{base: {{box: [-0.01,0.01,-0.01,0.01,-0.01,0.01]}}}}
constraints: [{{region: base, dofs: [1,1,1,1,1,1]}}]
solver: {{backend: numpy, mode: dynamic, n_steps: 10, total_time: 1.0e-3}}
output: {{monitors: {{}}}}
""", encoding="utf-8")
    return str(yp)


def _rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_a_bent_section_fractures_column_by_column():
    """Fig. 8's partially fractured state, with this law driving it: the
    outermost fibres reach `eps_uf` first and FAC climbs a column at a time
    while the member is still standing.

    Both columns go, not just the tensile one, and that is the fibre layout of
    v0.1 rather than the material: the curvature of a Hermite element under
    one end rotation changes sign along the member, so each column is in
    tension at one Gauss station -- and a fibre broken at ANY station is
    broken (Cai et al. eq. 20).
    """
    import tempfile
    import sys

    sys.path.insert(0, "tests")
    from test_fracture import _solver

    with tempfile.TemporaryDirectory() as td:
        s = _solver(_fibre_bar(td))
        mass = float(s.mass.sum())
        seen = {}
        for theta in (0.05, 0.10, 0.35):
            s.orient[1] = _rot_z(theta)
            s.q = np.zeros(s.ndof)
            _, actions = s.internal_force()
            seen[theta] = (float(s.fracture.fac[0]), float(actions[0, 5]))
        assert seen[0.05][0] == 0.0
        assert seen[0.10][0] == pytest.approx(1 / 3)
        assert seen[0.35][0] == pytest.approx(2 / 3)
        assert not s.fracture.latch.fractured[0], "a partly broken section is not severed"
        # the section still carries moment. The second return of
        # `internal_force` is the section resultant at the largest station,
        # SIGNED since v1.0 (it was its magnitude, +11933.4; now
        # -11933.4 = -integral of sigma y dA, the same sign as the internal
        # force +f_j at the end the station is nearer). The end actions the
        # failure criteria read did not change.
        assert abs(seen[0.05][1]) > 0.0
        assert float(s.mass.sum()) == mass


def test_an_axial_pull_takes_every_fibre_and_deactivates_the_member():
    """The other end of the same pipeline: uniform strain, so all fifteen
    fibres reach the fracture strain together, FAC hits 1 and the element
    stops carrying load with its mass left where it was."""
    import tempfile
    import sys

    sys.path.insert(0, "tests")
    from test_fracture import _solver

    with tempfile.TemporaryDirectory() as td:
        s = _solver(_fibre_bar(td, name="axial"))
        mass = float(s.mass.sum())
        q = np.zeros(s.ndof)
        q[6] = 0.2 * 0.01                      # 1% axial strain: elastic-plastic
        s.q = q
        f, _ = s.internal_force()
        assert abs(f).max() > 1.0 and s.fracture.fac[0] == 0.0

        q[6] = 0.2 * 0.03                      # past eps_cri = 2%
        s.q = q
        s.internal_force()
        assert s.fracture.fac[0] == 1.0 and s.fracture.latch.fractured[0]
        f, actions = s.internal_force()
        assert abs(f).max() == 0.0
        assert actions[0].tolist() == [0.0] * 6
        assert float(s.mass.sum()) == mass


def _rc_fibre_bar(td, threshold=0.1, eps_cri=0.02, name="rc"):
    """Concrete cover + two bars: the section shape the FUSED KERNEL is for."""
    import pathlib

    msh = pathlib.Path(td) / f"{name}.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n"
                   "1 0 0 0\n2 1 0 0\n3 2 0 0\n$EndNodes\n$Elements\n2\n"
                   "1 1 2 1 1 1 2\n2 1 2 1 1 2 3\n$EndElements\n", encoding="utf-8")
    yp = pathlib.Path(td) / f"{name}.yaml"
    yp.write_text(f"""
name: {name}
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {str(msh).replace(chr(92), "/")}}}
materials:
  conc: {{type: concrete01, fpc: 3.0e7, epsc0: 0.002, fpcu: 6.0e6, epscu: 0.006,
          rho: 2400.0}}
  steel: {{type: steel_cyclic, grade: Q345B, E: 2.06e11, rho: 7850.0,
           eps_cri: {eps_cri}}}
sections:
  b: {{type: bar, material: steel, shape: general, area: 2.0e-4}}
  s: {{type: fiber, material: conc, shape: rect, width: 0.3, height: 0.3,
      ny: 4, nz: 4,
      bars: [{{y: 0.12, z: 0.12, section: b}}, {{y: -0.12, z: -0.12, section: b}}],
      fracture: {{threshold: {threshold}}}}}
assign: [{{region: all, section: s}}]
regions: {{base: {{box: [-0.01,0.01,-0.01,0.01,-0.01,0.01]}}}}
constraints: [{{region: base, dofs: [1,1,1,1,1,1]}}]
solver: {{backend: numpy, mode: dynamic, n_steps: 10, total_time: 1.0e-3}}
output: {{monitors: {{}}}}
""", encoding="utf-8")
    return str(yp)


def test_the_fused_kernel_now_has_a_law_that_can_fracture():
    """The gap v0.1 left open, closed.

    That stage wired deactivation through the fused fibre kernel and then had
    to test it with a hand-set mask, because none of the three laws the kernel
    can inline -- concrete01, concrete02, steel02 -- declared a fracture
    criterion. `steel_cyclic` compiles into the same kernel and does, so a
    section on the fast path now reaches FAC = 1 by itself.

    The two paths are asserted to agree EXACTLY, event included: the compiled
    law and the reference law are the same source (`_point.py_func`), so
    anything but a bit-for-bit match would mean the kernel is reading state
    the reference does not.
    """
    import tempfile
    import sys

    sys.path.insert(0, "tests")
    from test_fracture import _solver

    with tempfile.TemporaryDirectory() as td:
        yp = _rc_fibre_bar(td)
        fast = _solver(yp)
        if fast._kernel is None:
            pytest.skip(f"no fused kernel here: {fast.kernel_status}")
        assert "numba fused kernel" in fast.kernel_status
        assert "steel_cyclic" in fast.summary()
        assert "no criterion in Concrete01" in fast.summary()

        slow = _solver(yp)
        slow._kernel = None
        for eps in (0.001, 0.005, 0.015, 0.03):
            forces = []
            for s in (fast, slow):
                q = np.zeros(s.ndof)
                q[6], q[12] = eps, 2.0 * eps
                s.q = q
                forces.append(s.internal_force()[0].copy())
            assert np.array_equal(forces[0], forces[1]), eps
            assert fast.fracture.fac.tolist() == slow.fracture.fac.tolist()
        # two bars of eighteen fibres is FAC = 1/9, past the 0.1 threshold
        assert fast.fracture.fac[0] == pytest.approx(1 / 9)
        assert fast.fracture.n_fractured == 2 == slow.fracture.n_fractured


def test_the_compiled_law_is_its_own_python_source():
    """Not a twin that has to be pinned against a reference -- the same
    function object, compiled. `py_func` is the definition; if numba is not
    installed the definition is what runs."""
    if not hasattr(sc._point, "py_func"):
        pytest.skip("numba is not installed here")
    m = law()
    p = m.numba_params()
    a, b = m.init_state(1), m.init_state(1)
    eps = np.zeros((1, 6))
    path = np.concatenate([np.linspace(0, 0.03, 300), np.linspace(0.03, -0.02, 300),
                           np.linspace(-0.02, 0.05, 300), np.linspace(0.05, 0.40, 400)])
    for e in path:
        eps[0, 0] = e
        compiled = float(m.update(eps, a, 1.0)[0, 0])
        source = sc._point.py_func(float(e), b[0], p)
        assert compiled == source
    assert np.array_equal(a, b)
    assert m.numba_uniaxial_fn() is sc._point


# --------------------------------------------------------------------------- #
#  the elastic branch has to end
# --------------------------------------------------------------------------- #

def test_a_reversal_at_zero_stress_does_not_make_the_law_elastic_for_ever():
    """The defect the first tube member found, and the shape of it.

    Criterion 2 says unloading is elastic. Unloading STOPS being unloading the
    moment the stress changes sign, and the old test for that also demanded
    that the reversal had left a stress of the opposite sign to unload
    through. Every textbook reversal does; the one that matters does not. A
    point that turns round AT ZERO STRESS -- which in a dynamic run is every
    fibre, during the first oscillations, before anything has yielded -- then
    never left the elastic branch, and `sig = sig_rev + E (eps - eps_rev)` ran
    on with slope E past yield, past f_u, without limit.

    On Cai et al.'s Tube A that showed up as fibre stresses of 462 MPa on a
    law with f_y = 288 and f_u = 361, and a squash load of 147 kN against the
    paper's 100.8 kN.

    THE SEQUENCE THAT DOES IT is narrower than "a reversal", and the first
    version of this test missed it by starting the path at the first non-zero
    strain: evaluate the law ONCE AT ZERO -- which every solver does, because
    the internal force is assembled before anything has moved -- and the
    direction is set to +1 with nothing else recorded. The first step that
    goes the other way then registers a reversal at exactly (0, 0), and
    (0, 0) is the state the old test could not leave.

    So: one call at zero, then a monotone push. The answer must be the
    MONOTONIC curve, and must not depend on that first call having happened.

    THERE WERE TWO FAULTS, and the second was only visible once the first was
    fixed. (1) The elastic branch could not be left from (0, 0) at all, so the
    stress ran to 14 f_u. (2) With that fixed the point did leave -- onto the
    hysteresis SKELETON, as though it had already been cycled, when it had
    only been asked a question. The monotonic curve and the skeleton differ by
    10% in stress and by 2.5x in necking strain, so the point then fractured
    at a strain it should have carried. A step that does not move may not
    commit a loading direction either.
    """
    m = law()
    ey = m.fy / m.E
    st = m.init_state(1)
    eps = np.zeros((1, 6))
    assert float(m.update(eps, st, 1.0)[0, 0]) == 0.0     # the zero call
    assert st[0, 3] == 0.0, "a step that did not move committed a direction"
    for e in np.linspace(0.0, -3 * ey, 600)[1:]:
        eps[0, 0] = e
        after_zero = float(m.update(eps, st, 1.0)[0, 0])
    assert st[0, 2] == 0.0, "the first loading is monotonic, whatever its sign"

    straight = drive(m, [-3 * ey])[0][-1, 1]
    assert after_zero == pytest.approx(straight, rel=1e-15), "the zero call cost something"
    assert after_zero / m.fy == pytest.approx(-1.0, abs=1e-6), "the monotonic plateau"

    # and the size of a REAL excursion before the push may not matter either:
    # once the point has been loaded and reversed, it is on the skeleton, and
    # how far it went first does not change where the skeleton is
    answers = [drive(m, [tiny * ey, -3 * ey])[0][-1, 1] / m.fy
               for tiny in (1e-6, 1e-3, 0.01, 0.1, 0.5, 1.5)]
    assert max(answers) - min(answers) < 1e-9, answers
    assert answers[0] == pytest.approx(-1.0977, abs=1e-3)


def test_no_path_gets_past_the_ultimate_strength():
    """The general form of the same question, and the one worth keeping: a
    uniaxial law may harden cyclically, but it may not exceed the strength it
    was given, whatever the history that led there."""
    m = law()
    ey = m.fy / m.E
    paths = ([-20 * ey],
             [1e-6 * ey, -20 * ey],
             [0.5 * ey, -0.5 * ey, 20 * ey],
             [2 * ey, -2 * ey, 2 * ey, -2 * ey, 30 * ey],
             [1e-9 * ey, -1e-9 * ey, 1e-9 * ey, -25 * ey])
    for path in paths:
        rows, _st = drive(m, path)
        worst = np.abs(rows[:, 1]).max()
        assert worst <= m.fu * (1 + 1e-9), (path, worst / m.fu)

    # including the one that starts with the solver's evaluation at zero
    st = m.init_state(1)
    eps = np.zeros((1, 6))
    m.update(eps, st, 1.0)
    worst = 0.0
    for e in np.linspace(0.0, -20 * ey, 2000)[1:]:
        eps[0, 0] = e
        worst = max(worst, abs(float(m.update(eps, st, 1.0)[0, 0])))
    assert worst <= m.fu * (1 + 1e-9), worst / m.fu
