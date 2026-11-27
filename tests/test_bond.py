"""The MC2010 bond stress-slip law: envelope, reversals, and the three twins.

WHY IT EXISTS
Embedded bars here are perfectly bonded, and that was measured to be what set
the limit of the solid RC path: the drift a column survived scaled with the bar
SEGMENT length (32.0 / 18.0 / 11.0 mm for 100 / 50 / 25 mm segments) because a
crack of opening w put strain w/L into whatever segment crossed it. This law is
the length scale that was missing -- the transfer length comes out of tau(s) and
the bar perimeter rather than out of a discretisation choice.

This file tests the LAW only. It touches no element and no solver, so it cannot
break an existing case, and it is the whole of stage A in v0.1.
"""
import numpy as np
import pytest

from vfife.materials import create
from vfife.base.compat import trapezoid

FC = 25.6e6                      # Tanaka specimen 1, the case this serves


def _bond(**kw):
    kw.setdefault("fc", FC)
    return create("bond_ceb", **kw)


def _tau(mat, slips):
    """Walk a slip path, returning the bond stress at each point."""
    state = mat.init_state(1)
    out = []
    for s in np.atleast_1d(slips):
        x = np.zeros((1, 6))
        x[0, 0] = float(s)
        out.append(float(mat.update(x, state, 1.0)[0, 0]))
    return np.array(out), state


# --------------------------------------------------------------------------- #
#  The published envelope
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("condition,factor,s1,s2", [
    ("good", 2.5, 1.0e-3, 2.0e-3),
    ("poor", 1.25, 1.8e-3, 3.6e-3),
])
def test_the_envelope_is_the_one_model_code_2010_publishes(condition, factor, s1, s2):
    """tau_max = k sqrt(fc) with fc in MPa, and the two slip corners.

    The unit conversion is the part worth pinning: MC2010 writes the formula in
    MPa and everything here is in Pa, so getting it wrong is a factor of a
    thousand and would read as a bond that is not there.
    """
    m = _bond(condition=condition)
    assert m.tau_max == pytest.approx(factor * np.sqrt(FC / 1e6) * 1e6, rel=1e-12)
    assert m.s1 == pytest.approx(s1)
    assert m.s2 == pytest.approx(s2)
    assert m.tau_f == pytest.approx(0.4 * m.tau_max, rel=1e-12)


def test_the_four_branches_are_where_they_should_be():
    m = _bond(condition="good", s3=10.0e-3)
    e = lambda mm: float(m.envelope(np.array([mm * 1e-3]))[0])
    assert e(1.0) == pytest.approx(m.tau_max, rel=1e-12)      # end of the rise
    assert e(1.5) == pytest.approx(m.tau_max, rel=1e-12)      # plateau
    assert e(2.0) == pytest.approx(m.tau_max, rel=1e-12)      # end of plateau
    assert e(10.0) == pytest.approx(m.tau_f, rel=1e-12)       # end of descent
    assert e(50.0) == pytest.approx(m.tau_f, rel=1e-12)       # residual holds
    # the ascending branch is a power law with alpha = 0.4, so it is CONCAVE:
    # half the slip gives much more than half the stress, which is the whole
    # reason bond is stiff at small slip
    assert e(0.5) / e(1.0) == pytest.approx(0.5 ** 0.4, rel=1e-9)


def test_the_law_is_odd():
    """A bar pushed in and a bar pulled out meet the same ribs."""
    m = _bond()
    for mm in (0.3, 1.0, 3.0, 20.0):
        pos, _ = _tau(_bond(), [mm * 1e-3])
        neg, _ = _tau(_bond(), [-mm * 1e-3])
        assert pos[0] == pytest.approx(-neg[0], rel=1e-12)


def test_the_envelope_never_rises_after_the_plateau():
    m = _bond(s3=10.0e-3)
    a = np.linspace(2.0e-3, 30.0e-3, 200)
    env = m.envelope(a)
    assert np.all(np.diff(env) <= 1e-9)


# --------------------------------------------------------------------------- #
#  Reversals
# --------------------------------------------------------------------------- #

def test_unloading_leaves_the_envelope_and_is_stiffer_than_it():
    """Coming back from 3 mm, the first step must drop far faster than the
    envelope does -- otherwise the model would be reversing along the curve,
    which stores no energy and gives no hysteresis."""
    m = _bond(s3=10.0e-3)
    path = list(np.linspace(0, 3.0e-3, 60)) + [3.0e-3 - 1.0e-5]
    tau, _ = _tau(m, path)
    drop = tau[-2] - tau[-1]
    env_drop = float(m.envelope(np.array([3.0e-3]))[0]
                     - m.envelope(np.array([3.0e-3 - 1.0e-5]))[0])
    assert drop > 10 * abs(env_drop)


def test_the_largest_excursion_only_grows():
    """`s_max` is what stops a reversal re-entering the envelope from inside;
    if it could shrink the law would heal."""
    m = _bond()
    _, state = _tau(m, list(np.linspace(0, 4.0e-3, 80))
                    + list(np.linspace(4.0e-3, -1.0e-3, 80)))
    assert state[0, 0] == pytest.approx(4.0e-3, rel=1e-9)


def test_a_reversal_stays_inside_the_envelope_reached_so_far():
    """The unloading line is bounded both ways. Without the bound a stiff
    reload would shoot past the curve and invent strength.

    Two bounds, and the first version of this test used the wrong one. The
    envelope at the PEAK SLIP is not the ceiling: a path that rises to 3 mm has
    already crossed the plateau at 1-2 mm, where tau is tau_max = 12.65 MPa,
    while the envelope at 3 mm is on the descending branch at 11.70. Nothing
    was violated -- the test was comparing the whole path against a bound that
    only applies AFTER the peak.
    """
    m = _bond(s3=10.0e-3)
    peak = 3.0e-3
    rise = list(np.linspace(0, peak, 80))
    back = list(np.linspace(peak, -peak, 160)) + list(np.linspace(-peak, peak, 160))
    tau, _ = _tau(m, rise + back)

    # globally, nothing may exceed the peak of the law itself
    assert np.max(np.abs(tau)) <= m.tau_max * (1 + 1e-9)
    # and once the excursion has been made, the reversal is capped by the
    # envelope AT that excursion, which is lower
    bound = float(m.envelope(np.array([peak]))[0])
    assert bound < m.tau_max                       # 3 mm is past the plateau
    assert np.max(np.abs(tau[len(rise):])) <= bound * (1 + 1e-9)


def test_a_closed_slip_cycle_does_not_create_energy():
    """Round trip back to zero slip: the dissipated work must be >= 0. A sign
    error in the unloading rule shows up here as a loop that pumps energy, and
    that is what destroys an explicit run rather than merely biasing it."""
    m = _bond(s3=10.0e-3)
    peak = 2.5e-3
    path = np.concatenate([np.linspace(0, peak, 400),
                           np.linspace(peak, -peak, 800),
                           np.linspace(-peak, 0, 400)])
    tau, _ = _tau(m, path)
    work = float(trapezoid(tau, path))
    assert work >= -1e-6 * m.tau_max * peak


# --------------------------------------------------------------------------- #
#  The three implementations
# --------------------------------------------------------------------------- #

def test_the_numba_twin_follows_the_reference_through_a_cyclic_path():
    pytest.importorskip("numba")
    ref, fast = _bond(s3=10.0e-3), _bond(s3=10.0e-3)
    s_ref, s_fast = ref.init_state(1), fast.init_state(1)
    point = fast.numba_point_fn()
    path = np.concatenate([np.linspace(0, 3e-3, 200),
                           np.linspace(3e-3, -2e-3, 300),
                           np.linspace(-2e-3, 12e-3, 400),
                           np.linspace(12e-3, 0, 200)])
    worst = 0.0
    for s in path:
        x = np.zeros((1, 6))
        x[0, 0] = s
        a = float(ref.update(x, s_ref, 1.0)[0, 0])
        out = np.zeros(6)
        point(np.array([s, 0.0, 0.0, 0.0, 0.0, 0.0]), s_fast[0], 1.0, 1.0, out)
        worst = max(worst, abs(a - out[0]) / max(abs(a), 1.0e3))
    assert worst < 1e-12, f"reference and numba split by {worst:.2e}"


def test_the_warp_parameter_block_carries_every_constant_the_law_uses():
    """Seven numbers, in the order kernels/warp_materials.py indexes them. A
    params array that lost one would give the GPU a different law, and the
    symptom would be a backend disagreement rather than an error."""
    m = _bond(condition="poor", s3=12.0e-3)
    prm, _tab = m.warp_params()
    assert len(prm) == 9
    for index, attr in ((0, "tau_max"), (1, "s1"), (2, "s2"), (3, "s3"),
                        (4, "alpha"), (5, "tau_f"), (6, "k_unload"),
                        (7, "s_elastic"), (8, "k_initial")):
        assert prm[index] == pytest.approx(getattr(m, attr)), attr
    # the indices are also declared on the class, so a twin that reads prm[7]
    # and a params array that puts s_elastic somewhere else cannot both be
    # right without this noticing
    for attr, index in m.P.items():
        assert prm[index] == pytest.approx(getattr(m, attr)), attr


# --------------------------------------------------------------------------- #
#  Refusals
# --------------------------------------------------------------------------- #

def test_it_needs_either_a_concrete_strength_or_an_explicit_peak():
    with pytest.raises(ValueError, match="tau_max"):
        create("bond_ceb")


def test_an_unknown_bond_condition_is_refused():
    with pytest.raises(ValueError, match="good"):
        _bond(condition="excellent")


@pytest.mark.parametrize("kw", [
    {"s1": 2.0e-3, "s2": 1.0e-3},        # corners out of order
    {"alpha": 0.0},
    {"alpha": 1.5},
    {"tau_f_ratio": 1.5},                # friction above the peak
])
def test_an_inconsistent_envelope_is_refused(kw):
    with pytest.raises(ValueError):
        _bond(**kw)


def test_it_is_not_offered_to_elements_as_a_bar_or_solid_law():
    """It is a traction-separation law: the "strain" is a slip in metres and
    the "stress" is a bond stress. Nothing that expects stress from strain may
    pick it up by accident."""
    from vfife.elements import library

    for name, cap in library.LINE_ELEMENTS.items():
        assert "bond_ceb" not in cap.compatible_materials, name


def test_erosion_cannot_be_configured_against_it():
    """`damage_measure` returns None, which is how the solver refuses to run
    erosion on a material with no notion of failure. An interface does not
    erode; the bar and the concrete do."""
    assert _bond().damage_measure(_bond().init_state(1)) is None


# --------------------------------------------------------------------------- #
#  The splitting failure mode
#
#  These test the SHAPE, not the coefficient. MC2010 Table 6.1-1 supplies the
#  number in front of (fcm/20)**0.25 and it was NOT verified against the code,
#  so it lives in `tau_max_coeff` and is documented as a value to check. Pinning
#  an unverified number in a test would turn a guess into a fact, which is the
#  failure this project has logged a dozen times.
# --------------------------------------------------------------------------- #

def test_splitting_is_weaker_than_pull_out():
    """The whole reason the mode matters. Modelling a thin-cover member as
    pull-out overestimates its bond, and that is the unsafe direction."""
    po = _bond()
    sp = _bond(failure_mode="splitting")
    assert sp.tau_max < po.tau_max


def test_splitting_has_no_plateau():
    """A split cover turns the curve over at the peak: s2 = s1 by construction,
    so there is no range of slip at constant bond."""
    sp = _bond(failure_mode="splitting")
    assert sp.s2 == sp.s1
    just_past = float(sp.envelope(np.array([sp.s1 * 1.05]))[0])
    assert just_past < sp.tau_max


def test_an_unconfined_split_leaves_no_friction():
    """A pulled-out bar still rubs on the concrete cylinder around it. A bar in
    split concrete has nothing left to rub against, so tau_f is zero -- and
    that is what makes splitting a cliff rather than a slope."""
    sp = _bond(failure_mode="splitting", confinement="unconfined")
    assert sp.tau_f == 0.0
    assert float(sp.envelope(np.array([30.0e-3]))[0]) == 0.0


def test_stirrups_leave_friction_behind_after_a_split():
    """Transverse steel holds the split cover together, so some bond survives."""
    sp = _bond(failure_mode="splitting", confinement="stirrups")
    assert sp.tau_f > 0.0
    assert float(sp.envelope(np.array([30.0e-3]))[0]) == pytest.approx(sp.tau_f)


def test_splitting_unloads_over_a_shorter_slip_than_pull_out():
    po = _bond()
    sp = _bond(failure_mode="splitting")
    assert sp.s3 < po.s3


def test_the_splitting_peak_scales_as_the_fourth_root_not_the_square_root():
    """PO goes as sqrt(fc), SP as (fcm/20)**0.25 -- a different exponent, so
    doubling the concrete strength moves them by different amounts. Getting the
    exponent wrong would be invisible at one strength and wrong at every other.
    """
    a = _bond(failure_mode="splitting", fc=20.0e6).tau_max
    b = _bond(failure_mode="splitting", fc=80.0e6).tau_max
    assert b / a == pytest.approx(4.0 ** 0.25, rel=1e-9)

    c = _bond(fc=20.0e6).tau_max
    d = _bond(fc=80.0e6).tau_max
    assert d / c == pytest.approx(4.0 ** 0.5, rel=1e-9)


def test_the_unverified_coefficient_can_be_supplied():
    """It defaults to a value the docstring marks as unchecked, so the way to
    use this properly is to pass the table's number."""
    m = _bond(failure_mode="splitting", tau_max_coeff=6.5)
    assert m.tau_max_coeff == 6.5
    ref = _bond(failure_mode="splitting", tau_max_coeff=13.0)
    assert ref.tau_max == pytest.approx(2.0 * m.tau_max, rel=1e-12)


def test_an_explicit_peak_overrides_the_formula_entirely():
    m = _bond(failure_mode="splitting", tau_max=4.0e6)
    assert m.tau_max == pytest.approx(4.0e6)


def test_the_default_mode_stays_pull_out():
    """Changing it would silently re-rate every model already in the repo."""
    assert _bond().failure_mode == "pull_out"


@pytest.mark.parametrize("bad", [
    {"failure_mode": "spalling"},
    {"confinement": "hoops"},
])
def test_an_unknown_mode_or_confinement_is_refused(bad):
    with pytest.raises(ValueError):
        _bond(**bad)


def test_the_splitting_branch_reaches_the_compiled_twin_unchanged():
    """The twins read the same seven numbers whatever produced them, so the new
    mode must not need a new code path -- but it does need proving, because a
    zero tau_f exercises the descending branch differently."""
    pytest.importorskip("numba")
    ref = _bond(failure_mode="splitting")
    fast = _bond(failure_mode="splitting")
    s_ref, s_fast = ref.init_state(1), fast.init_state(1)
    point = fast.numba_point_fn()
    path = np.concatenate([np.linspace(0, 8e-3, 400),
                           np.linspace(8e-3, -8e-3, 600),
                           np.linspace(-8e-3, 2e-3, 300)])
    worst = 0.0
    for s in path:
        x = np.zeros((1, 6))
        x[0, 0] = s
        a = float(ref.update(x, s_ref, 1.0)[0, 0])
        out = np.zeros(6)
        point(np.array([s, 0.0, 0.0, 0.0, 0.0, 0.0]), s_fast[0], 1.0, 1.0, out)
        worst = max(worst, abs(a - out[0]) / max(abs(a), 1.0e3))
    assert worst < 1e-12, f"splitting split the twins by {worst:.2e}"


# --------------------------------------------------------------------------- #
#  The origin
# --------------------------------------------------------------------------- #

def test_the_initial_tangent_is_finite():
    """alpha < 1 makes d(tau)/ds infinite at s = 0. On paper that is fine; in an
    explicit scheme it means an element cannot size a stable time step, and the
    bar dof ends up under-massed -- the same defect class that still carries an
    xfail in tests/test_mixed_mesh.py.

    Found by a rigid-body test on the element: a slip of 1e-16 was returning
    79 Pa, which is negligible physically and not negligible in an absolute
    tolerance."""
    m = _bond()
    assert np.isfinite(m.k_initial)
    assert m.E == m.k_initial                    # what an element asks for
    # the unloading slope IS the capped origin tangent since the reversal
    # jump found on case_12b (materials/bond.py); nothing is steeper
    assert m.k_initial >= m.k_unload


def test_the_cut_off_barely_moves_the_envelope():
    """It has to bound the tangent without changing the law anywhere that
    matters. s1/1000 caps the tangent at 63x the s1 secant and shifts the curve
    by less than a part in ten thousand of tau_max above the cut-off."""
    m = _bond()
    a = np.linspace(m.s_elastic, m.s1, 200)
    capped = m.envelope(a)
    pure = m.tau_max * (a / m.s1) ** m.alpha
    assert np.max(np.abs(capped - pure)) < 1e-4 * m.tau_max


def test_below_the_cut_off_the_law_is_linear():
    m = _bond()
    a = np.array([0.1, 0.3, 0.6]) * m.s_elastic
    tau = m.envelope(a)
    assert np.allclose(tau / a, m.k_initial, rtol=1e-12)


def test_the_cut_off_reaches_the_compiled_twin():
    pytest.importorskip("numba")
    ref, fast = _bond(), _bond()
    s_ref, s_fast = ref.init_state(1), fast.init_state(1)
    point = fast.numba_point_fn()
    worst = 0.0
    for s in np.linspace(0.0, 3.0 * ref.s_elastic, 200):
        x = np.zeros((1, 6))
        x[0, 0] = s
        a = float(ref.update(x, s_ref, 1.0)[0, 0])
        out = np.zeros(6)
        point(np.array([s, 0.0, 0.0, 0.0, 0.0, 0.0]), s_fast[0], 1.0, 1.0, out)
        worst = max(worst, abs(a - out[0]) / max(abs(a), 1.0))
    assert worst < 1e-12


@pytest.mark.parametrize("bad", [0.0, -1e-6, 2.0e-3])
def test_a_cut_off_outside_the_ascending_branch_is_refused(bad):
    with pytest.raises(ValueError, match="s_elastic"):
        _bond(s_elastic=bad)
