"""What a bearing is worth on the Hangzhou Bay unit.

Version 0.4 measured a cost it could not price: `case_24d`'s piers are
MONOLITHIC with the girder, because a bearing is a `connector2` and the
fibre-beam solver cannot host one. These are the checks behind the elastic
twin that prices it -- the reference frame first, because a reference solution
nobody has verified is not a reference.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))
import generate_hzb_girder as hz          # noqa: E402


# ------------------------------------------------------ the reference solve
def test_the_pricing_frame_is_itself_right():
    """On four vertical rollers and no piers the twin has to be the classical
    three-span continuous beam, whose middle span is
    `5 q L^4 / 384 EI - M_pier L^2 / 8 EI` with M_pier from the three-moment
    equation. 0.6% -- the rest is the 5 m discretisation."""
    assert hz._four_rollers() == pytest.approx(hz._four_rollers_exact(), rel=0.01)
    assert hz._four_rollers() < 0.0                       # sagging is negative


def test_a_rigid_link_reproduces_the_monolithic_joint():
    """The spring block has to become the connection it replaces as its
    stiffnesses go up. If it did not, every number below would be measuring
    the spring rather than the bearing."""
    mono = hz.bearing_price(None, hz.EPS_PRESTRESS)["mid"]
    rigid = hz.bearing_price([hz.RIGID_LINK] * 2, hz.EPS_PRESTRESS)["mid"]
    assert rigid == pytest.approx(mono, rel=0.05)


# ------------------------------------------------------------ what it buys
def _penalty(bearings):
    """What the prestress shortening costs the middle span, mm."""
    return (hz.bearing_price(bearings, hz.EPS_PRESTRESS)["mid"]
            - hz.bearing_price(bearings, 0.0)["mid"])


def test_the_monolithic_joint_costs_more_than_stage_95_said_it_did():
    """-12.4 mm on this elastic twin, and v0.4 read -9.88 on the fibre model.

    IT USED TO BE -7.15 HERE, and the change is not a fix to this file: it is
    the pier section. v0.4 had no published statement of it and derived a
    SOLID 6.25 x 1.80 m from the piers' weights; three papers state a HOLLOW
    6.25 x (2.60 -> 4.00), walls 0.50, which is 4.4x stiffer in the plane of
    this frame. A stiffer pier fights the girder's
    shortening harder, so the monolithic joint's penalty goes UP by 74% --
    v0.4's headline was an understatement, not an overstatement.
    """
    assert _penalty(None) == pytest.approx(-12.4, abs=0.6)


def test_the_bearing_removes_it_entirely():
    """-0.0005 mm against -12.4. Not reduced -- removed, and still removed
    after the pier section was corrected to the stated hollow one."""
    assert abs(_penalty([hz.POT_FIXED, hz.POT_SLIDING])) < 0.05
    assert abs(_penalty([hz.POT_FIXED] * 2)) < 0.05


def test_either_release_kills_it_and_the_sliding_one_is_not_the_important_one():
    """THE FINDING. The penalty is a MOMENT and THRUST transfer, so releasing
    the rotation alone removes it just as completely as releasing the slide --
    and a FIXED pot, which does not slide at all, fixes it. Sweeping one
    component at a time with the other two rigid, the transition sits near the
    pier's own EI/L (5.4e10 N.m/rad) for the rotation and an order lower for
    the longitudinal spring."""
    soft_rot = dict(hz.RIGID_LINK, kr=1.0e+7)
    soft_lon = dict(hz.RIGID_LINK, kx=1.0e+5)
    assert abs(_penalty([soft_rot] * 2)) < 0.05
    assert abs(_penalty([soft_lon] * 2)) < 0.05
    # and both are monotone in between, so the transition is not an artefact
    seq = [abs(_penalty([dict(hz.RIGID_LINK, kr=k)] * 2))
           for k in (1e14, 1e11, 1e10, 1e9, 1e7)]
    assert all(a >= b for a, b in zip(seq, seq[1:]))


def test_the_sensitive_assumption_moves_from_the_pier_to_the_bearing():
    """The other half of the finding, and the reason the case header sweeps it:
    the bearing's VERTICAL stiffness does nothing to the penalty and everything
    to the dead-load deflection. case_24d's assumed number (the pier wall) was
    worth 10%; this one is worth more than a factor of three."""
    pen = [abs(_penalty([dict(hz.POT_FIXED, kz=kz)] * 2))
           for kz in (1e14, 1e10, 2e9)]
    assert max(pen) < 0.05                       # untouched by kz
    dead = [hz.bearing_price([dict(hz.POT_FIXED, kz=kz)] * 2, 0.0)["mid"]
            for kz in (1e14, 2e9)]
    assert dead[1] / dead[0] > 3.0               # and dominated by it


def test_one_pier_has_to_be_fixed():
    """Both abutments are longitudinal rollers. If BOTH bearings slid freely
    the unit would have no longitudinal restraint at all -- a rigid-body mode,
    and the linear solve is singular. That is why a real unit fixes exactly
    one pier, and it is worth an assertion because the failure is silent in a
    relaxation solver (it converges to an arbitrary translation)."""
    free = dict(hz.POT_SLIDING, kx=0.0)
    with pytest.raises(np.linalg.LinAlgError):
        hz.bearing_price([free, free], 0.0)


# --------------------------------------------------- where the camber goes
#  The v0.4 contract claimed the middle spans agree "because the side span
#  is dominated by the drape, which only the fibre model has". That is wrong:
#  the drape is the SAME in all three spans -- M_PRESTRESS reads 151 MN.m at
#  every mid-span. These pin the real reason.
def test_the_prestress_moment_is_the_same_in_every_span():
    """If it were not, everything below would be a statement about the tendon
    layout rather than about continuity."""
    M = hz.M_PRESTRESS.reshape(3, 14)            # 3 spans x 14 elements
    assert M[0] == pytest.approx(M[2][::-1], rel=1e-9)   # and symmetric about x = 105
    # each span's own mid-span pair, and its maximum
    assert M[0][6:8].mean() == pytest.approx(M[1][6:8].mean(), rel=0.01)
    assert M.max(axis=1) == pytest.approx([M.max()] * 3, rel=0.01)


def test_a_free_curvature_needs_a_free_end():
    """THE MECHANISM, on one span with no load. A constant free curvature
    between two rotationally fixed ends produces EXACTLY no deflection -- the
    fixed-end moments cancel it -- and `kappa L^2 / 8` with the ends free."""
    k, L = 199.43e-6, 70.0
    assert hz.curvature_needs_a_free_end(False) == pytest.approx(1e3 * k * L * L / 8, rel=1e-3)
    assert abs(hz.curvature_needs_a_free_end(True)) < 1e-6


def test_the_camber_lands_in_the_side_spans_not_in_proportion_to_the_spans():
    """So the split is ~100:1 while the dead load's is the classical 10.4:1.
    The abutment is a hinge; the middle span's ends are held by the side
    spans, so it keeps almost none of its own camber."""
    c = hz.camber_split()
    assert c["camber_side"] > 50.0 and c["camber_mid"] < 1.0      # both UP
    assert c["camber_side"] / c["camber_mid"] > 50.0
    assert c["dead"]["side"] / c["dead"]["mid"] == pytest.approx(9.97, rel=0.05)


def test_the_twin_with_the_curvature_reproduces_the_fibre_model():
    """And this is what makes the paragraph above a measurement rather than a
    story: the same elastic twin, given case_24f's own primary prestress
    moment as a free curvature, lands within 10% of the fibre model on BOTH
    spans -- the one it already matched and the one it was out by a factor of
    four on."""
    c = hz.camber_split()
    assert c["full"]["mid"] == pytest.approx(-7.977, rel=0.12)      # case_24f
    assert c["full"]["side"] == pytest.approx(-21.113, rel=0.12)
    # ... whereas the axial-only twin, which is case_24e, is not close at all
    assert c["axial"]["side"] / -21.113 > 3.0
