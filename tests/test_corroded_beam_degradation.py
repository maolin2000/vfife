"""The degradation curve is the deliverable, so it is pinned.

`benchmarks/corroded_beam_degradation.py` is what 子课题1 shows for the
corrosion model: eight measured flexural capacities against eight computed
ones, over 0.01% to 8.31% of strand mass loss. With the transient beam model
ruled out by localisation and the bond work shelved (decision D3),
this section analysis is not a stopgap -- it is the answer -- and an answer
that nothing asserts is an answer that drifts.

The tests are on the SHAPE of the agreement rather than on eight numbers: the
mean error, that the ductility law is what buys it, and that the curve is
monotone. A change to any of the four corrosion laws that made the fit worse,
or that made the ductility law unnecessary, would trip one of them.
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from corroded_beam_degradation import moment_curvature          # noqa: E402
from generate_corroded_pc_beam import (                         # noqa: E402
    LI_BEAMS, corroded_strand)

PRESTRESS_0 = 7.786e+08


@pytest.fixture(scope="module")
def curve():
    """(rho, measured, area+strength only, with the rupture limit) x 8."""
    out = []
    for _nm, rho, m_meas in LI_BEAMS:
        c = corroded_strand(rho, PRESTRESS_0)
        r = moment_curvature(c)
        free = r[:, 1].max() / 1e+3
        alive = r[:, 2] < c["eps_u"]
        rupt = r[alive, 1].max() / 1e+3
        out.append((rho, m_meas, free, rupt))
    return np.array(out)


def test_the_full_model_tracks_the_eight_measured_capacities(curve):
    """Mean |error| 6.2%, worst 10.5% (A7), every beam under its measurement.

    Until v1.0 it was 4.1% and 9.9%. The strand law became OpenSees'
    Steel02, whose envelope stays under its hardening asymptote (the law
    before it ran 3.45% over), and all eight capacities moved down by the
    same 2.6-2.9%: an offset from the steel, not a change in how the
    corrosion laws track the mass loss."""
    e = 100.0 * (curve[:, 3] - curve[:, 1]) / curve[:, 1]
    assert np.abs(e).mean() < 7.0, f"mean |err| {np.abs(e).mean():.1f}%"
    assert np.abs(e).max() < 12.0, f"worst {np.abs(e).max():.1f}% on {e}"


def test_the_ductility_law_is_what_buys_the_agreement(curve):
    """Area and strength alone are not enough, and this is the measurement that
    says so: drop the rupture limit and the scatter of the errors more than
    doubles (6.3 against 2.7 points; 6.5 against 3.3 before v1.0).

    The SCATTER, not the worst error, since v1.0: the new steel02 moved
    every beam down by about 2.8%, which is an offset both variants share,
    and it brought the full model's worst (A7, -10.5%) close to the
    area-only model's (A5, +12.6%). What the ductility law does is the same
    as before -- it takes A5 from +12.6% to -2.0% and makes the curve fall
    monotonically -- and the spread is what measures that."""
    with_rupture = np.std(100.0 * (curve[:, 3] - curve[:, 1]) / curve[:, 1])
    area_only = np.std(100.0 * (curve[:, 2] - curve[:, 1]) / curve[:, 1])
    assert area_only > 1.5 * with_rupture, (
        f"area+strength alone scatters {area_only:.1f} points and the full model "
        f"{with_rupture:.1f}: the ductility law is no longer earning its place")


def test_area_and_strength_alone_give_a_curve_that_is_not_even_monotone(curve):
    """The qualitative half of the same point. Capacity must fall as the strand
    corrodes; without the ductility law one beam comes back UP, because the
    derived hardening ratio grows as the plastic range is squeezed and nothing
    stops the strand using it."""
    assert np.diff(curve[:, 2]).max() > 0.0, (
        "area+strength alone is now monotone -- if that is a real improvement, "
        "this test and the one above should be rewritten around it")
    assert np.all(np.diff(curve[:, 3]) <= 1e-9), (
        "the full model's capacity must fall monotonically with mass loss")


def test_the_failure_mode_switches_where_the_campaign_says_it_does(curve):
    """Li's beams go from concrete crushing to wire fracture at about 8% of
    mass loss. The model is not told that: it comes out of the ductility law
    meeting the section's own curvature demand."""
    modes = []
    for _nm, rho, _m in LI_BEAMS:
        c = corroded_strand(rho, PRESTRESS_0)
        r = moment_curvature(c)
        k = int(np.argmax(r[:, 1]))
        modes.append(r[k, 2] >= c["eps_u"])          # True = wire break
    assert not modes[0], "the uncorroded control must fail in compression"
    assert modes[-1], "the most corroded beam must break its wire"
    # and the switch happens once, not back and forth
    assert sum(a != b for a, b in zip(modes, modes[1:])) == 1, modes
