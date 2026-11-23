"""The selection guide, as assertions.

`docs/steel02.md` tells an analyst which of the four uniaxial steel laws to
use, and every claim in it is a MEASUREMENT: how round the reversal is, how
much the peak grows under constant-amplitude cycling, whether the law can
fracture, whether it has a compiled scalar twin. A table like that in a
document goes stale the first time a law changes; here it cannot, because the
same four numbers are computed from the laws themselves and the ordering the
document claims is what is asserted.

Timings are NOT asserted -- they belong to the machine, not to the library --
but the twin column is, because that is the one that decides whether a fibre
section runs on the fused kernel or falls back to NumPy at four times the cost.
"""

import numpy as np
import pytest

from vfife.materials.base import create

E, FY = 2.0e11, 4.0e8
AMPLITUDE = 0.01


def laws():
    return {
        "rebar": create("rebar", E=E, rho=7850.0, yield_strength=FY, hardening=0.01 * E),
        "steel02": create("steel02", E=E, rho=7850.0, Fy=FY, b=0.01),
        "steel_cyclic": create("steel_cyclic", grade="Q345B", E=2.06e11, rho=7850.0),
        "piecewise_uniaxial": create(
            "piecewise_uniaxial", E=E, rho=7850.0,
            tension=[[0.0, FY], [0.05, 1.2 * FY]],
            compression=[[0.0, FY], [0.05, 1.2 * FY]]),
    }


def _cycles(m, n=6, pts=400):
    st = m.init_state(1)
    eps = np.zeros((1, 6))
    rows, prev = [], 0.0
    for target in [AMPLITUDE, -AMPLITUDE] * n:
        for e in np.linspace(prev, target, pts)[1:]:
            eps[0, 0] = e
            rows.append((e, float(m.update(eps, st, 1.0)[0, 0])))
        prev = target
    return np.array(rows)


def _reversal_roundness(m):
    """The strain, in yield strains, over which the unloading tangent falls
    from 0.9 E to 0.05 E. Zero is a corner; a Menegotto-Pinto curve rounds it
    over about half a yield strain."""
    ey = getattr(m, "eps_y", FY / m.E)
    st = m.init_state(1)
    eps = np.zeros((1, 6))
    for e in np.linspace(0.0, AMPLITUDE, 600)[1:]:
        eps[0, 0] = e
        s_prev = float(m.update(eps, st, 1.0)[0, 0])
    e_prev, started = AMPLITUDE, None
    for e in np.linspace(AMPLITUDE, -AMPLITUDE, 20000)[1:]:
        eps[0, 0] = e
        s = float(m.update(eps, st, 1.0)[0, 0])
        k = (s - s_prev) / (e - e_prev) / m.E
        if started is None and k < 0.9:
            started = e
        elif started is not None and k < 0.05:
            return abs(started - e) / ey
        e_prev, s_prev = e, s
    return 0.0


def test_the_reversal_is_a_corner_for_two_of_them_and_rounded_for_two():
    r = {name: _reversal_roundness(m) for name, m in laws().items()}
    assert r["rebar"] < 0.05 and r["piecewise_uniaxial"] < 0.05
    # 4.61 eps_y since v1.0: OpenSees' R falls from R0 = 20 toward 2.3
    # after a 1% excursion (cR1 = 0.925). The law before it kept R near 19
    # and rounded over 0.58 eps_y -- the Bauschinger effect hardly showed.
    assert 3.5 < r["steel02"] < 6.0
    assert r["steel_cyclic"] > 2.0
    assert min(r["steel_cyclic"], r["steel02"]) > 10 * max(r["rebar"], r["piecewise_uniaxial"])


def test_two_of_them_keep_growing_under_constant_amplitude_cycling():
    """The trap the document names: `rebar` and `piecewise_uniaxial` harden
    isotropically without saturating, so six cycles at one amplitude raise the
    peak by a third and the energy per cycle by a fifth. In a time-history
    analysis that is hysteretic energy the structure does not have."""
    grown, stable = {}, {}
    for name, m in laws().items():
        r = _cycles(m)
        n = len(r) // 12
        peak = r[10 * n:, 1].max() / r[:2 * n, 1].max()
        w1 = abs(np.trapezoid(r[2 * n:4 * n, 1], r[2 * n:4 * n, 0]))
        w6 = abs(np.trapezoid(r[10 * n:12 * n, 1], r[10 * n:12 * n, 0]))
        (grown if peak > 1.3 else stable)[name] = (peak, w6 / w1)
    assert set(grown) == {"rebar", "piecewise_uniaxial"}
    assert set(stable) == {"steel02", "steel_cyclic"}
    for name, (_, energy) in grown.items():
        assert energy > 1.1, name
    for name, (_, energy) in stable.items():
        assert energy == pytest.approx(1.0, abs=0.01), name


def test_only_two_of_them_keep_a_fibre_section_on_the_fused_kernel():
    """`numba_uniaxial_fn` is not a detail: a section whose bar law has no
    scalar twin drops off `kernels/numba_fiber_beam` entirely and runs four
    times slower.

    WHAT IS ASSERTED IS THE DECLARATION, not the compiled object. The first
    version of this test asked whether the twin came back non-None, which is
    true on a developer machine and was false on CI -- the fast tier installed
    `.[dev,plot]` and no numba, so every law answered None and the test failed
    where nothing was wrong. A law either offers a scalar twin
    or it does not; whether numba is installed to compile it is the
    environment's business, and it is checked separately below.

    The fast tier now installs `cpu` as well (tests/test_ci_layers.py), so on
    CI the second half runs too. The skip stays for the machine that has no
    numba: a missing optional extra is not a broken library.
    """
    declared = {name: hasattr(m, "numba_uniaxial_fn") for name, m in laws().items()}
    assert declared == {"rebar": False, "steel02": True,
                        "steel_cyclic": True, "piecewise_uniaxial": False}

    numba = pytest.importorskip("numba", reason="no numba: nothing to compile")
    assert numba is not None
    built = {name: getattr(m, "numba_uniaxial_fn", lambda: None)() is not None
             for name, m in laws().items()}
    assert built == declared, "a declared twin that does not build"


def test_only_two_of_them_can_fracture_and_they_say_so_differently():
    """`rebar` fractures only when given a `rupture_strain` -- until then it
    declares None, which is the v0.1 contract: no criterion is not the same
    as a criterion that has not fired."""
    st = create("rebar", E=E, rho=7850.0, yield_strength=FY,
                hardening=0.01 * E).init_state(1)
    assert create("rebar", E=E, rho=7850.0, yield_strength=FY,
                  hardening=0.01 * E).failure(st) is None
    with_rupture = create("rebar", E=E, rho=7850.0, yield_strength=FY,
                          hardening=0.01 * E, rupture_strain=0.05)
    assert with_rupture.failure(with_rupture.init_state(1)) is not None
    for name, m in laws().items():
        declared = m.failure(m.init_state(1)) is not None
        assert declared == (name == "steel_cyclic"), name
