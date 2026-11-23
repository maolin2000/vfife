"""`steel02` can be told the strain at which the steel breaks.

WHY IT IS ON THIS LAW
    Four uniaxial steel laws, and until now only two could stop: `rebar` has
    `rupture_strain` but no fused fibre-beam kernel, and `steel_cyclic` has a
    degradation curve but wants a table of parameters. `steel02` is the one a
    prestressing strand is written with AND the one the fused kernel inlines,
    and it hardened for ever. A 1x7-15.2-1860 strand does not.

WHY THE MEASURE IS NOT `rebar`'s
    `rebar` latches on ACCUMULATED PLASTIC STRAIN, which grows on every
    reversal. This law has no plastic strain to accumulate -- Menegotto-Pinto
    rounds the knee and carries no yield surface -- so it latches on the PEAK
    TENSILE STRAIN the point has seen, which is what a tension test reports
    (A_gt on a mill certificate). The two agree for a tendon pulled once to
    failure and disagree for a bar being cycled. That is deliberate, the
    docstrings say so, and `test_the_two_laws_measure_different_things` below
    pins it so nobody "fixes" it by making them the same.
"""
import numpy as np
import pytest

from vfife.materials.base import create

E, FY, B = 1.95e11, 1.834e9, 0.008
RUP = 0.034


def _mat(**kw):
    return create("steel02", E=E, Fy=FY, b=B, R0=20.0, rho=7850, **kw)


def _pull(material, path):
    """Drive one point along `path`, return the stresses."""
    st = material.init_state(1)
    out = []
    for eps in path:
        out.append(float(material.update(np.array([[eps]]), st, 1.0)[0, 0]))
    return np.array(out), st


def test_without_a_rupture_strain_nothing_changes():
    """The eighth state column exists either way; it must not do anything."""
    path = np.linspace(0.0, 0.06, 200)
    a, sa = _pull(_mat(), path)
    b, sb = _pull(_mat(rupture_strain=10.0), path)       # far beyond the path
    assert np.array_equal(a, b)
    assert a[-1] > FY, "the law should still be carrying at 6% without a limit"
    # and the peak-strain column tracked the path in both
    assert sa[0, 7] == pytest.approx(path[-1])
    assert sb[0, 7] == pytest.approx(path[-1])


def test_the_point_is_cut_from_the_call_after_its_strain_reached_the_limit():
    """Read against the ACCEPTED peak strain, before this call raises it.

    Until v0.4 the call that crossed the limit already returned zero. In a
    staged relaxation every iteration re-drives the point from the last
    converged state, so a TRANSIENT overshoot of the trial strain -- which the
    fictitious dynamics produce routinely -- cut the fibre inside the
    iteration, the section lost the strand, and the level converged on the
    broken branch although the intact one was also in equilibrium. Measured
    on Li's A7: the strand broke at 790 MPa of converged stress with 40 mm
    elements and 928 MPa with 80 mm ones, which read as mesh dependence; with
    the latch deferred, both meshes reach 1288 MPa at the same 39.90 kN m.
    In an explicit time history the deferral is one time step.
    """
    path = np.array([0.0, 0.01, 0.02, 0.030, RUP - 1e-4, RUP + 1e-4, 0.04])
    sig, st = _pull(_mat(rupture_strain=RUP), path)
    # on the hardening asymptote, 1872 MPa (1.91 GPa before v1.0, when the
    # envelope ran 3.45% over it)
    assert sig[4] > FY, "still carrying just below the limit"
    assert sig[5] > FY, "the crossing call is not cut: its history is not accepted yet"
    assert sig[6] == 0.0
    assert st[0, 7] >= RUP


def test_a_trial_redriven_from_the_committed_state_is_never_cut():
    """What a staged relaxation does: copy the committed state back, evaluate
    a trial strain. However far past the limit the trial goes, the committed
    peak strain is below it, so the fibre carries; only once a state beyond
    the limit is COMMITTED is it cut."""
    m = _mat(rupture_strain=RUP)
    committed = m.init_state(1)
    m.update(np.array([[0.030]]), committed, 1.0)
    for trial in (RUP + 1e-4, 0.05, 0.10):
        st = committed.copy()
        assert float(m.update(np.array([[trial]]), st, 1.0)[0, 0]) > FY
    st = committed.copy()
    m.update(np.array([[RUP + 1e-4]]), st, 1.0)      # converged past the limit
    committed = st                                    # ...and committed
    st = committed.copy()
    assert float(m.update(np.array([[RUP + 2e-4]]), st, 1.0)[0, 0]) == 0.0
    assert m.failure(committed)[0] == 1.0


def test_rupture_does_not_heal_when_the_beam_unloads():
    """`eps_max` only ever takes a maximum, so the latch cannot come back."""
    path = np.concatenate([np.linspace(0.0, 0.04, 60), np.linspace(0.04, 0.0, 60),
                           np.linspace(0.0, 0.02, 30)])
    sig, _ = _pull(_mat(rupture_strain=RUP), path)
    # index 0 is the unloaded start, which also reads zero -- skip it
    zeros = np.flatnonzero(sig[1:] == 0.0) + 1
    assert len(zeros), "the point never ruptured"
    cut = int(zeros[0])
    assert (sig[cut:] == 0.0).all(), "a ruptured point started carrying again"


def test_compression_never_ruptures_it():
    """A strand pushed to -5% is not broken; only tension counts."""
    path = np.concatenate([np.linspace(0.0, -0.05, 80), np.linspace(-0.05, 0.0, 80)])
    sig, st = _pull(_mat(rupture_strain=0.01), path)
    assert (sig != 0.0).any()
    assert st[0, 7] < 0.01
    assert abs(sig).max() > FY * 0.9


def test_the_two_laws_measure_different_things_and_that_is_on_purpose():
    """Twenty small cycles: `rebar` accumulates to its limit, `steel02` does not.

    Both are given 2% as the number. The path never exceeds 1% of total
    strain, so a peak-strain law must survive it; an accumulated-plastic-strain
    law must not.
    """
    path = []
    for _ in range(20):
        path += list(np.linspace(0.010, -0.010, 40))
        path += list(np.linspace(-0.010, 0.010, 40))
    path = np.array(path)

    s2, _ = _pull(_mat(rupture_strain=0.02), path)
    assert s2[-1] != 0.0, "steel02 must survive: its peak tensile strain stayed 1%"

    rb = create("rebar", E=E, rho=7850, yield_strength=FY, hardening=B * E,
                rupture_strain=0.02)
    rs, _ = _pull(rb, path)
    assert rs[-1] == 0.0, "rebar must rupture: it accumulates plastic strain"


def test_the_compiled_kernel_and_the_readable_loop_agree_across_the_rupture():
    """The fused fibre-beam kernel inlines the compiled point law, so a latch
    that exists in only one of the two would be invisible until a case ran."""
    material = _mat(rupture_strain=RUP)
    path = np.concatenate([np.linspace(0.0, 0.02, 60), np.linspace(0.02, -0.01, 40),
                           np.linspace(-0.01, 0.05, 120), np.linspace(0.05, 0.0, 40)])
    n = 16
    fast_state = material.init_state(n)
    ref_state = material.init_state(n)
    strain = np.zeros((n, 6))
    worst = 0.0
    for eps in path:
        strain[:, 0] = eps * (1.0 + 0.02 * np.arange(n) / n)
        fast = material.update(strain, fast_state, 1.0)[:, 0].copy()
        ref = material._update_reference(
            strain, ref_state, np.zeros((n, 6)))[:, 0].copy()
        worst = max(worst, float(np.max(np.abs(fast - ref))))
    assert worst == 0.0, f"compiled and reference disagree by {worst:.3e} Pa"
    # the spread means the fibres cross the limit at different times
    assert 0 < (fast_state[:, 7] >= RUP).sum() <= n


def test_the_element_above_can_see_which_fibres_are_gone():
    """A zero stress is also what an unloaded elastic fibre reports, so the
    element counts `failure`, not stresses."""
    material = _mat(rupture_strain=RUP)
    st = material.init_state(3)
    strain = np.zeros((3, 6))
    strain[:, 0] = [0.0, 0.02, 0.05]
    material.update(strain, st, 1.0)
    f = material.failure(st)
    assert list(f) == [0.0, 0.0, 1.0]
    d = material.damage_class(st)
    assert d[2] == material.DAMAGE_FAILED
    assert d[0] != material.DAMAGE_FAILED and d[1] != material.DAMAGE_FAILED


def test_a_law_without_the_limit_reports_no_failure_measure_at_all():
    material = _mat()
    assert material.failure(material.init_state(2)) is None


def test_a_negative_rupture_strain_is_refused():
    with pytest.raises(ValueError, match="rupture_strain must be positive"):
        _mat(rupture_strain=-0.01)
    with pytest.raises(ValueError, match="rupture_strain must be positive"):
        _mat(rupture_strain=0.0)
