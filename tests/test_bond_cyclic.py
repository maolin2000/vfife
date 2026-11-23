"""`bond_cyclic`: the MC2010 envelope with degradation and a pinched reversal.

THE TEST THAT MATTERS MOST IS THE FIRST ONE.
    A monotonic path must come out of this law bit for bit identical to
    `bond_ceb`. Everything the class adds is switched on by cycling and by
    nothing else, so if a monotonic push moves at all, the damage measure is
    counting something it should not -- and every pushover result already
    produced (the Tanaka column) would quietly change the day
    someone swapped the material name. That assertion is the reason the energy
    accumulator banks only what is dissipated INSIDE the envelope.

WHAT IS DELIBERATELY NOT ASSERTED
    That the degradation matches Eligehausen's measured loops. It cannot be:
    `damage_coeff` and `damage_exponent` are exposed defaults to calibrate,
    not citations, and the energy measure they act on is this class's own.
    What IS asserted is everything that must hold for ANY calibration -- the
    damage is monotone, bounded, dissipative, never re-strengthening, and
    identical across the backends.
"""
import numpy as np
import pytest

from vfife.materials import create

FC = 25.6e6


def _cyc(**kw):
    return create("bond_cyclic", fc=FC, **kw)


def _mono(**kw):
    return create("bond_ceb", fc=FC, **kw)


def _run(material, path, state=None, commit=True):
    """Drive a single point along a slip history; return the stresses.

    Every point is COMMITTED by default, which is what makes the history a
    load path: `update` alone is one explicit iteration and banks nothing.
    Pass `commit=False` to drive the material the way a solver's inner loop
    does, which is what `test_the_iteration_path_cannot_bank_damage` needs.
    """
    state = material.init_state(1) if state is None else state
    out = np.empty(len(path))
    x = np.zeros((1, 6))
    for i, s in enumerate(path):
        x[0, 0] = s
        out[i] = material.update(x, state, 1.0)[0, 0]
        if commit and hasattr(material, "commit"):
            material.commit(state)
    return out, state


def _cycles(amplitude, n, per=400):
    """n full symmetric cycles at one amplitude, starting and ending at zero."""
    legs = [np.linspace(0.0, amplitude, per)]
    for _ in range(n):
        legs.append(np.linspace(amplitude, -amplitude, 2 * per))
        legs.append(np.linspace(-amplitude, amplitude, 2 * per))
    legs.append(np.linspace(amplitude, 0.0, per))
    return np.concatenate(legs)


# --------------------------------------------------------------------------- #
#  The regression guard
# --------------------------------------------------------------------------- #

def test_a_monotonic_path_is_bit_for_bit_the_monotonic_law():
    """The whole point of adding this as a SECOND material rather than a flag.

    Bit for bit, not `approx`: the two evaluate the same expression, because
    on a path that only extends |s| the cyclic law's `g` is exactly 1.0 and
    its envelope branch IS bond_ceb's. A tolerance here would hide a damage
    term leaking into the monotonic answer at the sixth digit, which is
    exactly the failure this test exists to catch.
    """
    path = np.linspace(0.0, 15.0e-3, 3000)
    a, _ = _run(_mono(), path)
    b, st = _run(_cyc(), path)
    assert np.array_equal(a, b)
    assert st[0, 3] == 0.0, "a monotonic push banked cyclic energy"


def test_a_monotonic_path_in_the_other_direction_is_identical_too():
    """The law is odd, and the energy test above only pushed one way."""
    path = np.linspace(0.0, -15.0e-3, 3000)
    a, _ = _run(_mono(), path)
    b, st = _run(_cyc(), path)
    assert np.array_equal(a, b)
    assert st[0, 3] == 0.0


def test_the_splitting_branch_is_inherited_unchanged():
    """`bond_cyclic` adds a cyclic rule, not a new envelope. If it had
    reimplemented the envelope, the splitting mode -- a different shape with a
    zero residual -- is where the copy would have drifted first."""
    path = np.linspace(0.0, 8.0e-3, 2000)
    a, _ = _run(_mono(failure_mode="splitting"), path)
    b, _ = _run(_cyc(failure_mode="splitting"), path)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
#  Damage: the part whose absence was called unsafe
# --------------------------------------------------------------------------- #

def test_damage_is_monotone_bounded_and_starts_at_zero():
    """Bounded by 1, not strictly below it: exp(-1.6 * 50**1.1) underflows to
    exactly zero in double precision, so total loss of bond IS reachable and
    is a legitimate end state -- the channel is gone and the bar is held only
    by its own truss segments and the transverse penalty. It takes about
    20 * E_ref of cyclic dissipation per unit bonded area to get there, which
    for this law is 1.8 MJ/m^2; nothing in a structural history goes near it.
    """
    m = _cyc()
    assert m.damage(0.0) == 0.0
    e = np.linspace(0.0, 50.0 * m.E_ref, 500)
    d = m.damage(e)
    assert np.all(np.diff(d) >= 0.0)
    assert np.all(d >= 0.0) and np.all(d <= 1.0)
    # the scale, pinned: one envelope's worth of cyclic energy costs 80%
    assert float(m.damage(m.E_ref)) == pytest.approx(1.0 - np.exp(-1.6),
                                                     rel=1e-12)


def test_energy_never_decreases_along_an_arbitrary_history():
    """Damage must not heal. The accumulator banks only positive increments,
    which is what makes that true without a second state column -- the same
    device `rebar.rupture_strain` uses on accumulated plastic strain."""
    m = _cyc()
    state = m.init_state(1)
    rng = np.random.default_rng(7)
    path = np.cumsum(rng.normal(0.0, 2.0e-4, 4000))
    banked = []
    x = np.zeros((1, 6))
    for s in path:
        x[0, 0] = s
        m.update(x, state, 1.0)
        m.commit(state)
        banked.append(float(state[0, 3]))
    banked = np.array(banked)
    assert np.all(np.diff(banked) >= 0.0)
    assert banked[-1] > 0.0, "a 4000-step random walk dissipated nothing"


def test_repeated_cycles_at_one_amplitude_degrade_the_peak():
    """The headline behaviour, and the one `bond_ceb` cannot produce at all:
    under a repeated excursion to the same slip the bond that comes back is
    weaker every time, and monotonically so."""
    m = _cyc()
    amp = 1.5e-3
    peaks = []
    state = m.init_state(1)
    x = np.zeros((1, 6))
    for _ in range(8):
        for s in np.concatenate([np.linspace(-amp, amp, 800),
                                 np.linspace(amp, -amp, 800)]):
            x[0, 0] = s
            tau = float(m.update(x, state, 1.0)[0, 0])
            m.commit(state)
        # the peak this cycle reached, read at the tip on the way through
        peaks.append(float(m.envelope(np.array([amp]))[0]
                           * (1.0 - m.damage(state[0, 3]))))
    peaks = np.array(peaks)
    assert np.all(np.diff(peaks) < 0.0), "peaks did not fall monotonically"
    assert peaks[-1] < 0.9 * peaks[0], (
        f"eight cycles cost only {(1 - peaks[-1] / peaks[0]) * 100:.1f}%")
    assert tau == tau  # the loop ran


def test_the_monotonic_law_shows_none_of_that():
    """A control. Without it, the previous test only shows that a number
    changes, not that this material is the reason."""
    m = _mono()
    amp = 1.5e-3
    state = m.init_state(1)
    x = np.zeros((1, 6))
    tips = []
    for _ in range(8):
        for s in np.concatenate([np.linspace(-amp, amp, 800),
                                 np.linspace(amp, -amp, 800)]):
            x[0, 0] = s
            tau = float(m.update(x, state, 1.0)[0, 0])
            if abs(s - amp) < 1e-12:
                tips.append(tau)
    assert max(tips) - min(tips) < 1e-6, "bond_ceb degraded, which it cannot"


def test_the_first_excursion_past_the_previous_maximum_rejoins_the_envelope():
    """Degraded, but on the curve: after cycling, pushing past s_max must give
    exactly (1-d) times the virgin envelope, not a value left over from the
    loop's cap."""
    m = _cyc()
    _tau, state = _run(m, _cycles(1.2e-3, 5))
    d = float(m.damage(state[0, 3]))
    assert d > 0.0
    push = np.linspace(0.0, 4.0e-3, 1500)
    tau, state = _run(m, push, state)
    expect = (1.0 - float(m.damage(state[0, 3]))) * float(
        m.envelope(np.array([4.0e-3]))[0])
    assert tau[-1] == pytest.approx(expect, rel=1e-12)


# --------------------------------------------------------------------------- #
#  The load path, and not the iteration path
# --------------------------------------------------------------------------- #

def test_the_iteration_path_cannot_bank_damage():
    """The defect that made the first version of this class useless.

    A relaxation solve reaches ONE equilibrium point by taking tens of
    thousands of explicit steps, and that approach oscillates. When `update`
    banked energy, every wiggle read as a reversal: on a monotonic tension
    case, changing only `n_steps` from 60 000 to 200 000 moved the slip from
    106.0 um to 24.3 um against a perfectly bonded 24.9 -- a factor of 4.4 in
    the answer from a solver setting.

    So: an arbitrarily violent path with no commits must bank exactly nothing.
    """
    m = _cyc()
    rng = np.random.default_rng(3)
    wiggle = np.cumsum(rng.normal(0.0, 5.0e-4, 3000))
    _tau, state = _run(m, wiggle, commit=False)
    assert state[0, 3] == 0.0
    assert float(m.damage(state[0, 3])) == 0.0


def test_solver_scale_wandering_moves_the_damage_not_at_all():
    """How much iteration-dependence is LEFT: none, and this is the measurement.

    AN EARLIER VERSION OF THIS TEST ASSERTED A RESIDUE AND EXPLAINED IT. It
    let `s_max` and the unloading anchor follow the iteration path, said that
    was deliberate ("so a monotonic solve keeps matching bond_ceb"), and
    bounded the damage change at 2% for 2% wander and 20% for 20%:

        sigma/increment    banked energy      change
             0                52705 J/m^2      0.00 %
             0.05             51102           -3.04
             0.20             46172          -12.40
             0.80             25417          -51.78

    The premise was wrong. The anchor following the iteration path is not a
    harmless residue: on case_12b it left a symmetric beam 71.7% asymmetric
    after its first load step, and a bar started 20 um off in a rigid host at
    300 um of permanent slip (tests/test_bonded_convergence.py). The solver
    now RESTORES every bond state to its committed copy before every force
    evaluation (core/staging.py, TrialState), so an iteration leaves nothing
    behind at all, and this test walks the path the way the solver does.

    With the unloading slope at k_initial the same wander that used to move
    the damage 3% moved it 74% through the old anchor -- a steeper line turns
    a small wander into a large stress excursion. Restored, it moves it 0%.
    """
    path = _cycles(1.5e-3, 3)[::40]          # a coarse load path
    increment = float(np.abs(np.diff(path)).mean())
    _direct, sd = _run(_cyc(), path)
    assert sd[0, 3] > 0.0, "the control path banked nothing to compare"

    def wander(fraction):
        m = _cyc()
        state = m.init_state(1)
        committed = state.copy()
        x = np.zeros((1, 6))
        rng = np.random.default_rng(5)
        for s in path:
            # 30 uncommitted iterations approaching the level, each one
            # restored to the committed point first, as the solver does;
            # then the converged value, and one commit
            for w in s + rng.normal(0.0, fraction * increment, 30):
                state[...] = committed
                x[0, 0] = w
                m.update(x, state, 1.0)
            state[...] = committed
            x[0, 0] = s
            m.update(x, state, 1.0)
            m.commit(state)
            committed[...] = state
        return abs(state[0, 3] - sd[0, 3]) / sd[0, 3]

    assert wander(0.02) == 0.0
    assert wander(0.20) == 0.0
    assert wander(2.00) == 0.0, "a wander larger than the step itself still leaves no trace"


def test_a_load_path_walked_with_no_wandering_banks_exactly_the_same():
    """The other half: with the iterations sitting exactly on the level, the
    number of them makes no difference at all. That is the property the
    per-iteration version could not have."""
    path = _cycles(1.5e-3, 3)[::40]
    _d, sd = _run(_cyc(), path)

    m = _cyc()
    state = m.init_state(1)
    x = np.zeros((1, 6))
    for s in path:
        for _ in range(50):                  # 50 iterations, same value
            x[0, 0] = s
            m.update(x, state, 1.0)
        m.commit(state)
    assert state[0, 3] == pytest.approx(sd[0, 3], rel=1e-12)


def test_bond_ceb_has_no_commit_so_the_solver_hook_leaves_it_alone():
    """The hook is a getattr, and this is what makes that safe: every result
    already produced with perfect bond or the monotonic law is untouched by
    the whole mechanism."""
    assert not hasattr(_mono(), "commit")
    assert hasattr(_cyc(), "commit")


# --------------------------------------------------------------------------- #
#  Pinching
# --------------------------------------------------------------------------- #

def test_the_reversal_slides_at_friction_rather_than_on_the_envelope():
    """The loop is pinched. Coming back through zero slip after a large
    excursion, the bar is sliding in a channel it already made, so the stress
    there is the friction plateau -- not the envelope value the monotonic law
    would hand back.

    THE SIGN IS NEGATIVE, AND THAT IS THE ONLY ADMISSIBLE ONE.
        The first version of this test asserted a POSITIVE plateau -- the bar
        had been pulled forward, so surely the bond still pulls back on it.
        Wrong: friction opposes the MOTION, the motion here is towards
        negative slip, and tau*ds is the dissipation. A positive tau while ds
        is negative would make tau*ds negative, which is an interface
        GENERATING energy. The sign is settled by thermodynamics before any
        appeal to the measured loops, and the monotonic control below reaches
        the same side of zero.

    The degradation factor is read at the instant of the crossing, not at the
    end of the leg: the rest of the reversal banks more energy, and comparing
    against the final `g` would be comparing against a bond that had not
    happened yet.
    """
    m = _cyc()
    amp = 4.0e-3
    path = np.concatenate([np.linspace(0.0, amp, 1200),
                           np.linspace(amp, -amp, 2400)])
    state = m.init_state(1)
    x = np.zeros((1, 6))
    at_zero = g_then = None
    seen_tip = False
    for s in path:
        x[0, 0] = s
        tau = float(m.update(x, state, 1.0)[0, 0])
        seen_tip = seen_tip or s >= amp - 1e-12
        if seen_tip and at_zero is None and s <= 0.0:
            at_zero, g_then = tau, 1.0 - float(m.damage(state[0, 3]))

    assert at_zero == pytest.approx(-g_then * m.tau_fric0, rel=0.02), (
        f"reversal at zero slip gave {at_zero / 1e6:.2f} MPa, not the "
        f"{-g_then * m.tau_fric0 / 1e6:.2f} MPa friction plateau")
    assert abs(at_zero) < 0.6 * m.tau_max


def test_the_monotonic_law_has_no_friction_to_slide_on():
    """The control for the pinching test: at zero slip `bond_cyclic` sits on
    its friction level; `bond_ceb` has no such level and must not.

    THIS TEST USED TO ASSERT THE OPPOSITE SHAPE -- "bond_ceb unloads on a
    straight line all the way to the opposite envelope, so at zero slip it is
    well past the friction level", a FAT loop that the pinched law was said
    to replace. That fat loop was an artefact of the s1-secant unloading
    slope, 63x softer than the loading tangent, which made unloading sit
    BELOW loading and a small cycle gain energy (materials/bond.py). With the
    unloading slope at k_initial the monotonic law drops to the envelope at
    the current slip almost vertically, which at zero slip is zero: no
    friction, no fat loop, and the two laws still differ where the pinching
    test looks, for the right reason now.
    """
    m = _mono()
    amp = 4.0e-3
    _t, state = _run(m, np.linspace(0.0, amp, 1200))
    back = np.linspace(amp, -amp, 2400)
    tau_back, _state = _run(m, back, state)
    at_zero = float(tau_back[np.argmin(np.abs(back))])
    assert abs(at_zero) < 0.1 * m.tau_max, (
        f"bond_ceb carries {at_zero / 1e6:.2f} MPa at zero slip; it has no "
        "friction level to carry it on")


def test_the_loop_meets_the_envelope_at_the_tip_without_a_jump():
    """The cap is written so that it equals the envelope exactly at +-s_max,
    which is what keeps the force continuous when a reload arrives at the far
    tip. A discontinuity there is a step change in nodal force, and in an
    explicit scheme that is a shock, not a rounding error."""
    m = _cyc()
    amp = 3.0e-3
    _t, state = _run(m, np.linspace(0.0, amp, 1000))
    path = np.linspace(amp, -amp, 4000)          # reload onto the far tip
    tau, state = _run(m, path, state)
    # the largest step in stress anywhere on the reload leg, against the
    # largest step the ENVELOPE itself takes over the same slip increment
    ds = abs(path[1] - path[0])
    jump = float(np.abs(np.diff(tau)).max())
    assert jump < 3.0 * m.k_unload * ds, (
        f"a step of {jump / 1e6:.3f} MPa over {ds * 1e6:.2f} um is a jump, "
        "not a branch change")


def test_pinching_can_be_switched_off_and_the_loop_fattens():
    """`friction_ratio = 0` leaves nothing to slide on, so the reversal falls
    to zero stress in the middle. Asserting that the knob does what its name
    says is what stops it becoming decorative."""
    m = _cyc(friction_ratio=0.0)
    amp = 4.0e-3
    _t, state = _run(m, np.linspace(0.0, amp, 1200))
    back = np.linspace(amp, -amp, 2400)
    tau_back, _state = _run(m, back, state)
    assert abs(float(tau_back[np.argmin(np.abs(back))])) < 1e-6


# --------------------------------------------------------------------------- #
#  Thermodynamics, and the stability bound
# --------------------------------------------------------------------------- #

def test_a_closed_loop_dissipates_and_does_not_generate_energy():
    """The one property no calibration is allowed to break. Integrated round a
    closed cycle the work must be positive -- the interface takes energy out
    of the model and never puts it in. A sign error in the cap, or an
    unloading line steeper on the way back than the way out, shows up here as
    a negative area, and in a solver as a run that gains kinetic energy from
    nothing (which is precisely how the first bonded element was caught)."""
    m = _cyc()
    amp = 2.0e-3
    path = np.concatenate([np.linspace(0.0, amp, 1000),
                           np.linspace(amp, -amp, 2000),
                           np.linspace(-amp, amp, 2000)])
    tau, _state = _run(m, path)
    # the closed part: from the first arrival at +amp to the second
    start = 999
    loop_s, loop_t = path[start:], tau[start:]
    area = float(np.trapezoid(loop_t, loop_s)) if hasattr(np, "trapezoid") \
        else float(np.trapz(loop_t, loop_s))
    assert area > 0.0, f"the loop generated {-area:.3e} J/m^2"


def test_the_law_never_exceeds_the_virgin_envelope():
    """Degradation only takes away. A returned stress above what an UNDAMAGED
    bar could ever have offered would mean the model had invented bond, which
    is the unsafe direction of the error this class exists to remove.

    AGAINST WHAT, EXACTLY. The first version of this test compared against the
    virgin envelope at the CURRENT slip and failed, correctly and for a reason
    worth keeping: a bar that has slipped 4 mm and is sliding back through
    zero carries friction, ~4.2 MPa, while the virgin envelope at 0.002 mm is
    0.98 MPa. Residual friction does not vanish because the slip reading is
    momentarily small. The bound that actually says "invented nothing" is the
    envelope at the larger of the current slip and the largest excursion so
    far -- which is where all the bond this point ever had came from.
    """
    m = _cyc()
    state = m.init_state(1)
    rng = np.random.default_rng(11)
    path = np.cumsum(rng.normal(0.0, 3.0e-4, 5000))
    x = np.zeros((1, 6))
    worst = 0.0
    for s in path:
        x[0, 0] = s
        tau = abs(float(m.update(x, state, 1.0)[0, 0]))
        reach = float(m.envelope(np.array([max(abs(s), state[0, 0])]))[0])
        worst = max(worst, tau - reach)
    assert worst <= 1e-6, f"exceeded the virgin envelope by {worst:.3e} Pa"


def test_the_stability_bound_the_class_claims_actually_bounds_the_slopes():
    """`self.E` is what an element asks for when sizing its time step, and the
    class asserts in prose that damage and the reload ramp cannot beat the
    inherited `k_initial`. Under-massing a dof is the defect that still
    carries an xfail in tests/test_mixed_mesh.py, so the claim is measured
    here rather than believed."""
    m = _cyc()
    path = np.concatenate([np.linspace(0.0, 5.0e-3, 4000),
                           np.linspace(5.0e-3, -5.0e-3, 8000),
                           np.linspace(-5.0e-3, 5.0e-3, 8000),
                           np.linspace(5.0e-3, -1.0e-3, 4000)])
    tau, _state = _run(m, path)
    # the legs share their endpoints, so ds is zero at each junction and the
    # secant there is 0/0. Those steps are not a slope, they are the same
    # point twice; the first version of this test reported nan and asserted
    # nan <= E, which is False and looked like a real failure.
    ds = np.diff(path)
    keep = ds != 0.0
    slope = np.abs(np.diff(tau)[keep] / ds[keep])
    assert slope.max() <= m.E * (1.0 + 1e-9), (
        f"a secant of {slope.max():.3e} beats the declared bound "
        f"{m.E:.3e} Pa/m")


def test_the_response_is_odd_in_the_whole_history():
    """Flip every slip and every stress must flip. The cap is written
    symmetrically to make this true; an asymmetry would mean a column pushed
    left and a column pushed right disagreed for no physical reason."""
    m1, m2 = _cyc(), _cyc()
    path = np.concatenate([np.linspace(0.0, 3.0e-3, 800),
                           np.linspace(3.0e-3, -2.0e-3, 1000),
                           np.linspace(-2.0e-3, 5.0e-3, 1200)])
    a, sa = _run(m1, path)
    b, sb = _run(m2, -path)
    assert np.abs(a + b).max() < 1e-9 * m1.tau_max
    assert sa[0, 3] == pytest.approx(sb[0, 3], rel=1e-12)


# --------------------------------------------------------------------------- #
#  The three implementations
# --------------------------------------------------------------------------- #

def test_the_numba_twin_follows_the_reference_through_a_cyclic_path():
    """Not a formality: the cyclic law's state carries an ACCUMULATOR, so a
    twin that differs by one ulp per step does not stay within one ulp -- the
    damage diverges, and with it the envelope both branches are measured
    against. This path banks real energy before it is compared."""
    pytest.importorskip("numba")
    ref, fast = _cyc(), _cyc()
    s_ref, s_fast = ref.init_state(1), fast.init_state(1)
    point = fast.numba_point_fn()
    path = np.concatenate([np.linspace(0, 3e-3, 200),
                           np.linspace(3e-3, -3e-3, 400),
                           np.linspace(-3e-3, 3e-3, 400),
                           np.linspace(3e-3, -3e-3, 400),
                           np.linspace(-3e-3, 12e-3, 400),
                           np.linspace(12e-3, 0, 200)])
    worst = 0.0
    x = np.zeros((1, 6))
    scratch = np.zeros(6)
    for s in path:
        x[0, 0] = s
        a = float(ref.update(x, s_ref, 1.0)[0, 0])
        point(np.array([s, 0.0, 0.0, 0.0, 0.0, 0.0]), s_fast[0], 1.0, 1.0,
              scratch)
        worst = max(worst, abs(a - scratch[0]) / max(abs(a), 1.0e3))
    assert worst < 1e-12, f"reference and numba split by {worst:.2e}"
    # and the accumulator itself, which is what the stresses above were
    # measured against -- two runs can agree on tau while drifting in damage
    assert s_fast[0, 3] == pytest.approx(s_ref[0, 3], rel=1e-12)


def test_the_warp_parameter_block_carries_every_constant_the_law_uses():
    """Fourteen numbers: bond_ceb's nine plus the five the cyclic rule needs.
    A params array that lost one would hand the GPU a different law, and the
    symptom would be a backend disagreement rather than an error."""
    m = _cyc(condition="poor", damage_coeff=2.0, s_reload=0.5e-3)
    prm, _tab = m.warp_params()
    assert len(prm) == 14
    for attr, index in m.P.items():
        assert prm[index] == pytest.approx(getattr(m, attr)), attr
    # the first nine must still mean what the monotonic law's twin reads, or
    # the shared `bond_ceb_envelope` device function would be given garbage
    for attr, index in create("bond_ceb", fc=FC).P.items():
        assert m.P[attr] == index, attr


# --------------------------------------------------------------------------- #
#  Refusals
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kw, match", [
    ({"damage_coeff": -1.0}, "damage_coeff"),
    ({"damage_exponent": 0.0}, "damage_exponent"),
    ({"friction_ratio": 1.5}, "friction_ratio"),
    ({"s_reload": 0.0}, "s_reload"),
])
def test_nonsense_parameters_are_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        _cyc(**kw)


def test_it_still_refuses_what_the_monotonic_law_refuses():
    """The inherited validation has to survive the subclass's __init__, which
    calls super() FIRST for exactly this reason."""
    with pytest.raises(ValueError, match="fc|tau_max"):
        create("bond_cyclic")
    with pytest.raises(ValueError, match="condition"):
        _cyc(condition="excellent")


def test_the_contract_names_the_cyclic_caveat():
    """The catalogue is what an analyst reads before choosing a law, and the
    thing they most need to know here is that four of its numbers are
    defaults to calibrate rather than published constants."""
    from vfife.materials import definition

    item = definition("bond_cyclic")
    assert item.kinematics == ("bond",)
    assert "calibrate" in item.caveat
    assert "bond_damage" in item.output_fields
