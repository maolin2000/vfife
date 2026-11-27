"""What the Confederation Bridge reconstruction has to answer for.

Every number in case_28* traces back to a published one. These are the checks
that say so: the section against Butcher's gross-section table, the haunch
against Dilger's soffit slope, the mass against Ghali's erection weight, the
trucks against their stated totals, and the frame against the frequencies two
other groups published for it. The cross-checking plane frame in
benchmarks/generate_confederation.py is itself checked here first -- a
reference solution nobody has verified is not a reference.
"""
import pathlib
import sys

import numpy as np
import pytest

from vfife.api import _with_element, prepare_case
from vfife.core.structural import make_structural_solver

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))
import generate_confederation as cb          # noqa: E402
from vfife.base.compat import trapezoid

CASES = pathlib.Path(__file__).resolve().parents[1] / "cases"


def _solver(path):
    prep = prepare_case(str(path))
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


# --------------------------------------------------------- the reference solve
def test_the_cross_checking_frame_is_itself_right():
    """The plane frame the cases are compared against, on two problems whose
    answer is known: a cantilever's tip under a point load, and the support
    moment a temperature gradient raises on a two-span beam (3 EI kappa / 2,
    the closed form case_27 is built on)."""
    E, A, I, L, P = 2.1e+11, 0.1, 9.7e-4, 10.0, 1.0e+4
    fr = cb.Frame()
    n = [fr.node(x, 0.0) for x in np.linspace(0.0, L, 21)]
    for k in range(20):
        fr.element(n[k], n[k + 1], E * A, E * I)
    u = fr.solve({3 * n[-1] + 1: -P}, [3 * n[0], 3 * n[0] + 1, 3 * n[0] + 2])
    assert -u[3 * n[-1] + 1] == pytest.approx(P * L ** 3 / (3 * E * I), rel=1e-9)

    kappa = 1.2e-5 * 15.0
    fr = cb.Frame()
    n = [fr.node(x, 0.0) for x in np.linspace(0.0, 2 * L, 41)]
    for k in range(40):
        fr.element(n[k], n[k + 1], E * A, E * I, (0.0, kappa))
    u = fr.solve({}, [3 * n[0], 3 * n[0] + 1, 3 * n[20] + 1, 3 * n[-1] + 1])
    # UP, and a quarter of the kappa L^2 / 8 a single span would bow: the
    # middle support is the whole point of case_27. The sign matters more than
    # it looks -- writing the free curvature the other way round passes every
    # check a UNIFORM temperature can make, because a uniform rise has no
    # curvature, and it was a disagreement with the solver on case_28b that
    # caught it here.
    assert u[3 * n[10] + 1] == pytest.approx(kappa * L * L / 32.0, rel=1e-9)


# --------------------------------------------------------------- the section
@pytest.mark.parametrize("depth, area, centroid, inertia", [
    (13990.0, 22.247193e+6, 6752.0, 5.7779e+14),     # Butcher 2009 App. E,
    (13000.0, 19.218202e+6, 6517.0, 4.5496e+14),     # west/east face of pier
])                                                   # and Sections 1/2
def test_the_reconstructed_section_lands_on_butchers_published_table(
        depth, area, centroid, inertia):
    """Two thicknesses are fitted on this table; area, centroid and I are
    then predictions, and all three land within 4%."""
    A, yc, I = cb.section(depth, n=40001)
    assert A == pytest.approx(area, rel=0.04)
    assert depth - yc == pytest.approx(centroid, rel=0.04)
    assert I == pytest.approx(inertia, rel=0.04)


def test_the_midspan_centroid_is_a_prediction_with_nothing_fitted():
    """At 4.5 m the bottom slab is its minimum 250 mm and the web its stated
    405: nothing is free, and the centroid comes out 2882 mm against the
    2908 Butcher publishes."""
    A, yc, I = cb.section(cb.D_MID, n=40001)
    assert cb.D_MID - yc == pytest.approx(2908.0, rel=0.01)


def test_the_haunch_band_is_not_the_void_band():
    """Between the deck's 250 mm at the centreline and its 650 over the web
    the VOID is inboard of the haunch and the deck outboard of it. Reading it
    the other way round leaves the area 3.4% high and the centroid 200 mm
    high -- invisible until something asks where the material is."""
    d, t_bot, t_web = 14000.0, 1102.0, 460.0
    y = 300.0                                     # inside the haunch band
    haunch = cb.HAUNCH_START + (y - cb.T_DECK_CL) / (cb.T_DECK_WEB - cb.T_DECK_CL) \
        * (cb.X_WEB_TOP_OUT - t_web - cb.HAUNCH_START)
    out, void = cb._edges(np.array([y]), d, t_bot, t_web)
    assert void[0] == pytest.approx(haunch)       # the void ends AT the haunch
    # the deck runs out to where its own soffit has dropped past y (the
    # cantilever thins from 650 mm at the web to 230 at the tip)
    assert out[0] == pytest.approx(np.interp(y, cb.CANT_T[::-1], cb.CANT_X[::-1]))
    # the wrong reading -- void between the haunch and the web -- would make
    # the half width out - x_web_inner + haunch instead of out - haunch, which
    # is 3.4% of area and 200 mm of centroid on the whole section
    xi = cb.X_WEB_TOP_OUT - cb.WEB_LEAN * y - t_web
    assert out[0] - void[0] == pytest.approx(out[0] - haunch)
    assert abs((out[0] - xi + haunch) - (out[0] - void[0])) > 800.0


def test_the_webs_carry_dilgers_published_line_load():
    """w_D2 = 0.248 MN/m is the self weight of the WEBS at the pier face (not
    of the girder, which is more than twice that) -- an independent check on
    the web thickness and the girder depth, from a different paper."""
    y = np.linspace(0.0, 13990.0, 40001)
    t_bot, t_web = cb.thickness(13990.0)
    w = cb._half_width(y, 13990.0, t_bot, t_web)
    band = (y > cb.T_DECK_WEB) & (y < 13990.0 - t_bot)
    A = 2 * trapezoid(w[band], y[band]) / 1e+6
    assert A * cb.RHO_C * 9.80665 / 1e+6 == pytest.approx(0.248, rel=0.12)


# ----------------------------------------------------------------- the haunch
def test_the_haunch_is_not_a_parabola():
    """The exponent falls out of two published depths (13 990 mm at the pier
    face, 13 000 mm seven metres away) and is 1.38, not 2. The check is
    Dilger's soffit slope: 1.38 gives 8.2 degrees against his 7.80, a parabola
    would give 11.2."""
    assert cb.P_HAUNCH == pytest.approx(1.38, abs=0.02)
    slope = np.degrees(np.arctan(
        cb.P_HAUNCH * cb.A_HAUNCH * (cb.CANT_HALF - 5000.0) ** (cb.P_HAUNCH - 1)))
    assert slope == pytest.approx(7.80, rel=0.06)
    assert cb.depth_at(0.0) == pytest.approx(cb.D_PIER)
    assert cb.depth_at(cb.CANT_HALF) == pytest.approx(cb.D_MID)


def test_the_box_outline_holds_85_percent_of_the_erection_weight():
    """Ghali gives 8000 t for a 192.5 m double cantilever. The outline
    integrates to 6800; the missing 1200 is the pier diaphragm, the deviator
    and anchor blocks and the tendons, which a beam model must carry as MASS
    and not as area -- putting them in the area would get I wrong too."""
    u = np.linspace(0.0, cb.CANT_HALF, 241)
    A = np.array([cb.section(v)[0] for v in cb.depth_at(u)])
    mass = 2 * trapezoid(A, u) / 1e+9 * cb.RHO_C / 1e+3
    assert mass == pytest.approx(6800.0, rel=0.02)
    assert mass * cb.MASS_FACTOR == pytest.approx(8000.0, rel=0.01)


# ----------------------------------------------------------------- the trucks
@pytest.mark.parametrize("name, total", [("blue", 379.0), ("grey", 346.0)])
def test_the_axles_sum_to_the_published_truck_weight(name, total):
    """Ghali's Fig. 15 labels five arrows with four numbers -- the middle
    label serves two axles. Reading it as four axles leaves the blue truck
    22 kN over its stated total and the grey 63 kN under; as five it is
    exactly 379 and exactly 346."""
    axles = cb.truck_axles(name)
    assert len(axles) == 5
    assert sum(p for _, p in axles) == pytest.approx(total * 1e+3, abs=1.0)
    # the offsets are measured from the group's own resultant, which is what
    # lets the two trucks share one `start` and stay in one cross section
    assert sum(o * p for o, p in axles) == pytest.approx(0.0, abs=1e-6)


def test_the_case_carries_the_axles_the_generator_computes():
    """The YAML is written by hand; this is the check that it still says what
    generate_confederation.truck_axles says."""
    import yaml
    case = yaml.safe_load((CASES / "case_28_confederation_trucks.yaml").read_text(encoding="utf-8"))
    got = [ld["moving_load"]["axles"] for ld in case["steps"][0]["loads"]]
    for axles, name in zip(got, ("blue", "grey")):
        for (o, p), (o_ref, p_ref) in zip(axles, cb.truck_axles(name)):
            assert o == pytest.approx(o_ref, abs=1e-3)
            assert p == pytest.approx(p_ref, rel=1e-3)


# ------------------------------------------------------------ the temperature
def test_the_published_profile_is_mostly_a_mean_rise():
    """Resolving Ghali's Fig. 21 profile on the real section: about 19-21 K of
    mean rise everywhere, and a gradient that is small where the girder is
    deep and four times larger where it is shallow -- because the same 9 K
    drop across the deck slab is a much sharper curvature on 4.5 m than on
    14 m."""
    T_pier, g_pier = cb.temperature_field(cb.D_PIER)
    T_mid, g_mid = cb.temperature_field(cb.D_MID)
    assert T_pier == pytest.approx(19.1, abs=0.3)
    assert T_mid == pytest.approx(21.0, abs=0.3)
    assert g_pier == pytest.approx(0.39, abs=0.03)
    assert g_mid == pytest.approx(1.74, abs=0.05)
    assert g_mid / g_pier > 4.0


def test_the_whole_profile_reproduces_ghalis_deflection_profile():
    """Fig. 21, digitised: +60.6 mm at midspan, +47.0 at x = -60, -43.7 at
    x = -180, -73.5 at the tips. The whole field lands on midspan and x = -180
    within 3%; the tips are 25-30% over, and that is the cantilever's own
    curvature reading a hot band Fig. 21 never dimensions."""
    mid, q, o, tip_w, tip_e = cb.thermal_deflection(1, 1)
    assert mid == pytest.approx(60.6, rel=0.06)
    assert o == pytest.approx(-43.7, rel=0.06)
    assert q == pytest.approx(47.0, rel=0.15)
    assert -70.0 > tip_w > -100.0


def test_what_the_gradient_of_that_profile_is_worth():
    """Nine millimetres at midspan, where the frame holds both ends of the
    span, and thirty-one at every tip, where 95 m of cantilever is free to
    curl. That difference is why case_28c exists."""
    whole = np.array(cb.thermal_deflection(1, 1))
    mean = np.array(cb.thermal_deflection(1, 0))
    grad = np.array(cb.thermal_deflection(0, 1))
    assert np.allclose(whole, mean + grad, atol=0.05)    # linear, as it must be
    assert grad[0] == pytest.approx(-8.8, abs=1.0)
    assert grad[3] == pytest.approx(-31.4, abs=2.0)
    assert grad[3] == pytest.approx(grad[4], abs=0.5)    # both tips alike


def test_the_tips_are_what_read_the_hot_band_at_the_deck():
    """The one thing Fig. 21 does not dimension is how deep the 27.3 -> 18.3
    drop goes. Taking it as the deck's 250 mm at the centreline instead of its
    650 over the web moves the tips 20% and midspan by less than half a
    millimetre -- so the tip is the measurement that would settle it, and the
    midspan agreement is not evidence either way."""
    thin = cb.thermal_deflection(1, 1, t_slab=250.0)
    thick = cb.thermal_deflection(1, 1, t_slab=650.0)
    assert abs(thin[0] - thick[0]) < 0.5
    assert thin[3] / thick[3] == pytest.approx(0.81, rel=0.05)


def test_the_uniform_rise_response_does_not_depend_on_E():
    """Every stiffness in a frame scales with E together, so a free-strain
    load that nothing external resists gives the same displacements whatever E
    is. That is what makes case_28c nearly a check on geometry and alpha alone
    -- and it is not true of case_28."""
    assert cb.thermal_deflection(1, 0, E=30.0e+9)[0] == pytest.approx(
        cb.thermal_deflection(1, 0, E=50.0e+9)[0], rel=1e-9)


# -------------------------------------------------------------------- the pier
def test_the_pier_wall_is_the_models_one_assumed_number():
    """No source states it. The truck deflection moves 25% over a fourfold
    change in it and the uniform-rise deflection barely 4% -- which is the
    honest ranking of what the two cases actually check."""
    t0 = cb.PIER_T
    try:
        truck, warm = {}, {}
        for f in (0.5, 1.0, 2.0):
            cb.PIER_T = t0 * f
            xs, w = cb.truck_deflection(0.0)
            truck[f] = 1e+3 * w[int(np.argmin(np.abs(xs)))]
            warm[f] = cb.thermal_deflection(1, 0)[0]
    finally:
        cb.PIER_T = t0
    assert truck[0.5] / truck[2.0] == pytest.approx(1.44, rel=0.12)
    assert truck[1.0] == pytest.approx(16.9, rel=0.05)
    # and the uniform rise barely notices: 4% over the same fourfold change
    assert warm[0.5] / warm[2.0] == pytest.approx(0.93, rel=0.04)


# ----------------------------------------------------------------- the modes
@pytest.mark.slow
def test_the_vertical_mode_matches_two_published_models():
    """Lau et al. (2004) Table 1 give 0.620 Hz for the in-plane mode of a
    frame like this one, from a shell model and a frame model that agree to
    1.3%; Lin et al. (2010) give a vertical period of 1.54 s (0.649 Hz). This
    model, with the girder mass raised 17.5% to reach Ghali's 8000 t, puts it
    at 0.609 -- inside 2% of one and 6% of the other, with no calibration.

    The other in-plane mode -- the frame swaying, which the PIERS govern --
    comes out at 0.590 against Lau's 0.476. That gap is the assumed pier wall,
    the same one case_28 is 25% sensitive to, and it is why the frequencies
    are a test and not a contract.
    """
    from vfife import modal
    s = _solver(CASES / "case_28b_confederation_thermal.yaml")
    res = modal.analyse(s, n_modes=4)
    f = np.asarray(res.frequencies)
    assert 0.54 < f[0] < 0.62
    assert f[1] == pytest.approx(0.620, rel=0.08)       # Lau Table 1
    assert f[1] == pytest.approx(0.649, rel=0.10)       # Lin Table 1


# ------------------------------------------- the superimposed dead load (28d)
def test_the_drop_in_beam_reaction_is_most_of_the_cantilever_tip():
    """Each 60 m drop-in beam is simply supported between two overhang tips,
    so under the same 54 kN/m each tip takes V = 1620 kN on top of the line
    load -- the V of Ghali's own Fig. 9. Leaving it out is not a correction,
    it is a different structure: the tips go from 89 mm to 28 and midspan the
    other way, from 66 to 104. It is also the only part of case_28d's load
    that is not written in the paper as a number."""
    with_v = cb.sdl_deflection(E=cb.E_SDL)
    without = cb.sdl_deflection(V=0.0, E=cb.E_SDL)
    assert with_v[0] == pytest.approx(65.8, rel=0.03)
    assert with_v[2] == pytest.approx(89.0, rel=0.03)
    assert without[0] > with_v[0]                  # midspan sags MORE without
    assert without[2] < 0.4 * with_v[2]            # and the tip barely moves


def test_the_superimposed_dead_load_reproduces_ghalis_own_two_numbers():
    """Ghali p. 1147: "about 70 mm at mid span and 80 mm at cantilever tips".
    Those are outputs of HIS analysis, so reproducing them needs HIS modulus:
    the measured Ec(28) = 38.7 GPa aged by the s = 0.25 he fitted to his own
    measured moduli, which at 400 days is 46.5. On the unaged 38.7 the same
    frame gives 79 and 107."""
    aged = cb.sdl_deflection(E=cb.E_SDL)
    unaged = cb.sdl_deflection(E=cb.E_C)
    assert aged[0] == pytest.approx(70.0, rel=0.10)
    assert 0.5 * (aged[2] + aged[3]) == pytest.approx(80.0, rel=0.15)
    assert unaged[0] / aged[0] == pytest.approx(cb.E_SDL / cb.E_C, rel=1e-6)


def test_this_load_is_a_check_on_the_girder_not_on_the_assumed_pier():
    """case_28's truck deflection moves 25% over a fourfold change in the
    unpublished pier wall. This one moves 3%, because it is nearly uniform and
    nearly symmetric and the piers mostly carry axial force. That is the whole
    argument for adding it: the reconstructed part is the part it tests."""
    t0 = cb.PIER_T
    try:
        out = {}
        for f in (0.5, 1.0, 2.0):
            cb.PIER_T = t0 * f
            out[f] = cb.sdl_deflection(E=cb.E_SDL)[0]
    finally:
        cb.PIER_T = t0
    assert abs(out[2.0] / out[0.5] - 1.0) < 0.05


def test_the_aged_modulus_is_derived_and_not_fitted():
    """E_SDL is Ec(28) x exp[s (1 - sqrt(28/t))] with Ghali's own s = 0.25
    applied to the MODULUS, and the age is the only assumption in it. A factor
    of four on the age is worth 5%, which is why the case can afford to assume
    it."""
    import math
    assert cb.E_SDL == pytest.approx(
        cb.E_C * math.exp(0.25 * (1.0 - math.sqrt(28.0 / cb.AGE_SDL))), rel=1e-12)
    young = cb.E_C * math.exp(0.25 * (1.0 - math.sqrt(28.0 / 200.0)))
    old = cb.E_C * math.exp(0.25 * (1.0 - math.sqrt(28.0 / 800.0)))
    assert old / young == pytest.approx(1.048, rel=0.01)
