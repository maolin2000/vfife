"""Post-tensioning friction: sigma(s) = sigma_con exp(-(mu theta + k s)).

WHAT CHANGED, AND WHY IT IS THE STEP TOWARDS POST-TENSIONING
    `prestress` used to be one number for a whole bar set -- the EFFECTIVE
    stress, with every loss but elastic shortening left to the analyst, which
    preflight said out loud on every prestressed case. A post-tensioned tendon
    does not have one stress: it loses force from the jack inwards, by
    curvature and by wobble, and then loses more near the live end as the
    wedges seat. So the prestrain became a value PER BAR ELEMENT, and with a
    `friction` block the solver computes the profile itself.

    That is also the piece post-tensioning could not be built without. The
    remaining gaps are staging (raising the prestress part-way through a run)
    and the unbonded tendon's single global force.

WHY IT NEEDED "A ROUTE IS ONE BAR" FIRST
    theta is the angle accumulated ALONG the tendon, and s is the arc length
    from the jack. Neither exists unless the tendon is one connected bar whose
    segments are laid in order. Drawn as touching pieces the profile would
    silently restart at each piece; that is refused by name.
"""
import math

import numpy as np
import pytest

from vfife.core.diagnostics import UserError
from vfife.core.solver import Solver
from vfife.elements import truss
from vfife.io.model import InputCase, ReinforcementGeometry

EP, AP, FPY = 195.0e9, 150.0e-6, 1860.0e6
SIG_CON = 1395.0e6                      # 0.75 f_pu of a 1860 MPa strand
MU, K = 0.25, 0.0015

L, SAG = 20.0, 0.60
PROFILE = [[0.0, 0.0, 0.80], [L / 2, 0.0, 0.80 - SAG], [L, 0.0, 0.80]]


def _tendon(segment_length=0.5, shape="parabola", points=None):
    entries = ReinforcementGeometry(
        path={"points": points or PROFILE, "shape": shape,
              "segment_length": segment_length}).expand("t")
    return truss.bars_from_spec({"lines": entries})


def _midpoints(nodes, conn):
    """(arc length to each element midpoint, accumulated turn before it)."""
    d = nodes[conn[:, 1]] - nodes[conn[:, 0]]
    Lg = np.linalg.norm(d, axis=1)
    t = d / Lg[:, None]
    s = np.cumsum(Lg) - 0.5 * Lg
    turn = np.arccos(np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1.0, 1.0))
    return s, np.concatenate([[0.0], np.cumsum(turn)])


# --------------------------------------------------------------------------- #
#  The law itself
# --------------------------------------------------------------------------- #

def test_the_profile_is_the_closed_form_element_by_element():
    nodes, conn = _tendon()
    sig = truss.tendon_profile(nodes, conn, SIG_CON, MU, K, jack="start")
    s, theta = _midpoints(nodes, conn)
    assert sig == pytest.approx(SIG_CON * np.exp(-(MU * theta + K * s)), rel=1e-12)
    # and it is a loss: monotone down from the jack, and a real one
    assert np.all(np.diff(sig) < 0.0)
    assert sig[0] == pytest.approx(SIG_CON, rel=1e-3)
    assert 0.90 < sig[-1] / SIG_CON < 0.96


def test_a_straight_tendon_loses_only_wobble():
    """theta is zero on a straight run, so exp(-k s) is the whole of it -- the
    check that the two terms are not entangled."""
    nodes, conn = _tendon(shape="polyline",
                          points=[[0.0, 0.0, 0.4], [L, 0.0, 0.4]])
    sig = truss.tendon_profile(nodes, conn, SIG_CON, MU, K)
    s, theta = _midpoints(nodes, conn)
    assert np.all(theta == 0.0)
    assert sig == pytest.approx(SIG_CON * np.exp(-K * s), rel=1e-12)


def test_the_curvature_term_matches_the_parabolas_own_angle():
    """A shallow parabola turns 4a/L at each end, so end to end it turns
    2 atan(4a/L) and the far-end loss is exp(-(mu*that + k*L)). This is the
    check that theta is the TENDON's angle and not an artefact of sampling:
    it holds on a mesh of 40 chords and on one of 200.
    """
    exact_theta = 2.0 * math.atan(4.0 * SAG / L)
    for seg in (0.5, 0.1):
        nodes, conn = _tendon(segment_length=seg)
        sig = truss.tendon_profile(nodes, conn, SIG_CON, MU, K)
        arc = float(np.linalg.norm(nodes[conn[:, 1]] - nodes[conn[:, 0]], axis=1).sum())
        far = SIG_CON * math.exp(-(MU * exact_theta + K * arc))
        assert sig[-1] == pytest.approx(far, rel=0.02)


def test_jacking_from_the_far_end_mirrors_the_profile():
    nodes, conn = _tendon()
    a = truss.tendon_profile(nodes, conn, SIG_CON, MU, K, jack="start")
    b = truss.tendon_profile(nodes, conn, SIG_CON, MU, K, jack="end")
    assert b == pytest.approx(a[::-1], rel=1e-12)


def test_jacking_from_both_ends_takes_the_smaller_loss_everywhere():
    """What jacking from both ends actually buys: each point gets whichever
    end is cheaper to reach, so the worst stress moves to midspan and rises."""
    nodes, conn = _tendon()
    one = truss.tendon_profile(nodes, conn, SIG_CON, MU, K, jack="start")
    two = truss.tendon_profile(nodes, conn, SIG_CON, MU, K, jack="both")
    assert np.all(two >= one - 1e-9)
    assert two.min() > one.min()
    assert two == pytest.approx(two[::-1], rel=1e-9)          # symmetric
    assert int(np.argmin(two)) in (len(two) // 2 - 1, len(two) // 2)


def test_jack_must_name_an_end():
    nodes, conn = _tendon()
    with pytest.raises(ValueError, match="start, end or both"):
        truss.tendon_profile(nodes, conn, SIG_CON, MU, K, jack="middle")


# --------------------------------------------------------------------------- #
#  Anchor set
# --------------------------------------------------------------------------- #

def test_anchor_set_reflects_the_profile_and_conserves_the_draw_in():
    """The wedges seat by `delta`, the tendon slips back at the live end, and
    friction acting the other way stops it after the draw-in length. What is
    asserted is the construction's own definition: the area between the two
    profiles, divided by E, IS the draw-in.
    """
    delta = 0.006
    nodes, conn = _tendon(segment_length=0.2)
    Lg = np.linalg.norm(nodes[conn[:, 1]] - nodes[conn[:, 0]], axis=1)
    before = truss.tendon_profile(nodes, conn, SIG_CON, MU, K)
    after = truss.tendon_profile(nodes, conn, SIG_CON, MU, K,
                                 anchor_set=delta, E=EP)

    lost = float(np.sum((before - after) * Lg)) / EP
    assert lost == pytest.approx(delta, rel=0.05)

    # the loss is at the LIVE end and nowhere else
    seated = after < before - 1e-6
    assert seated[0] and not seated[-1]
    assert np.all(np.diff(seated.astype(int)) <= 0), "the seated zone is one run"
    # ... and the peak stress is no longer at the anchorage
    assert after[0] < after.max()
    assert after[0] < before[0]


def test_too_much_draw_in_is_an_error_and_says_how_much_the_tendon_can_take():
    """A short or straight tendon cannot absorb much seating: reflecting the
    whole profile is all there is. Silently flattening it would understate the
    prestress everywhere, so it is refused with the number."""
    nodes, conn = _tendon(segment_length=0.2)
    with pytest.raises(ValueError, match="draws in more than this tendon"):
        truss.tendon_profile(nodes, conn, SIG_CON, MU, K, anchor_set=0.10, E=EP)


# --------------------------------------------------------------------------- #
#  Through the solver
# --------------------------------------------------------------------------- #

STRAND = {"type": "rebar", "E": EP, "rho": 7850.0, "yield_strength": FPY,
          "hardening": 2.0e9}


def _case(friction=None, prestress=SIG_CON, path=None, lines=None):
    assign = {"section": "t", "reinforcement": "t", "name": "t"}
    if prestress is not None:
        assign["prestress"] = prestress
    if friction is not None:
        assign["friction"] = friction
    return {
        "name": "pt", "mesh": {"dimensions": [L, 0.4, 1.0], "divisions": [20, 2, 5]},
        "materials": {"c": {"type": "linear_elastic", "E": 34.0e9, "nu": 0.2,
                            "rho": 2400.0}, "s": dict(STRAND)},
        "sections": {"b": {"type": "solid", "material": "c"},
                     "t": {"type": "bar", "material": "s", "shape": "circle",
                           "diameter": 0.0158, "area": AP}},
        "reinforcement": {"t": {"path": path or {
            "points": [[0.0, 0.2, 0.80], [L / 2, 0.2, 0.20], [L, 0.2, 0.80]],
            "shape": "parabola", "segment_length": 0.5}}},
        "assign": [{"section": "b", "region": "all"}, assign],
        "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 10},
        "output": {"path": "unused.h5"},
    }


def _solver(case):
    from vfife.cli import build
    from vfife.io import schema
    lowered = InputCase(**case).lower()
    c = schema.Case(**lowered)
    mesh, material, element = build(c, "unused.yaml")
    d = c.model_dump(mode="python")
    d["element"] = {**d.get("element", {}), "type": element}
    return Solver(d, mesh, material, log=lambda *_: None)


def test_the_solver_gives_every_segment_its_own_prestrain():
    s = _solver(_case(friction={"mu": MU, "k": K, "jack": "start"}))
    eps = np.asarray(s.rebar[0].pre.prestrain, dtype=float)
    assert eps.shape == (s.rebar[0].n_elem,)
    assert np.all(np.diff(eps) < 0.0)
    assert eps[0] * EP == pytest.approx(SIG_CON, rel=1e-3)
    # and the bar really carries it: the force at u = 0 is sigma * A per segment
    f, sig = s.rebar[0].force(np.zeros_like(s.u), 1.0, want_stress=True)
    axial = np.asarray(sig).reshape(len(eps), -1)[:, 0]
    assert axial == pytest.approx(eps * EP, rel=1e-9)


def test_without_friction_the_prestrain_is_still_one_number():
    """The old meaning is untouched: `prestress` alone is the effective stress
    the whole tendon carries, and nothing about that changed."""
    s = _solver(_case())
    assert s.rebar[0].pre.prestrain == pytest.approx(SIG_CON / EP)


def test_friction_without_prestress_is_refused():
    from vfife.io.model import InputCase
    with pytest.raises(ValueError, match="no `prestress`"):
        InputCase(**_case(friction={"mu": MU, "k": K}, prestress=None)).lower()


def test_a_jacking_stress_at_yield_is_still_refused_with_friction():
    """The guard reads the PEAK of the profile, which is at the jack."""
    with pytest.raises(UserError) as exc:
        _solver(_case(friction={"mu": MU, "k": K}, prestress=FPY * 1.01))
    assert exc.value.code == "PRESTRESS_YIELD"


def test_a_friction_profile_on_a_split_tendon_is_refused():
    """theta accumulates along a tendon, so the tendon has to BE one bar.
    Touching entries are separate bars and the profile would restart at each
    one -- a plausible wrong answer, which is the kind this project refuses."""
    from vfife.cli import build
    from vfife.io import schema
    case = _case(friction={"mu": MU, "k": K})
    lowered = InputCase(**case).lower()
    lowered["reinforcement"][0]["lines"] = [
        {"start": [0.0, 0.2, 0.8], "end": [L / 2, 0.2, 0.2], "segment_length": 0.5},
        {"start": [L / 2, 0.2, 0.2], "end": [L, 0.2, 0.8], "segment_length": 0.5},
    ]
    c = schema.Case(**lowered)
    mesh, material, element = build(c, "unused.yaml")
    d = c.model_dump(mode="python")
    d["element"] = {**d.get("element", {}), "type": element}
    with pytest.raises(UserError) as exc:
        Solver(d, mesh, material, log=lambda *_: None)
    assert exc.value.code == "FRICTION_SPLIT_TENDON"


def test_a_friction_profile_on_a_closed_hoop_is_refused():
    """Friction is measured from the jacking end and a loop has none."""
    case = _case(friction={"mu": MU, "k": K})
    case["reinforcement"] = {"t": {"hoop": {
        "corners": [[0.05, 0.05], [0.35, 0.95]], "axis": "x",
        "from": 1.0, "to": 3.0, "spacing": 1.0, "divisions": 3}}}
    with pytest.raises(UserError) as exc:
        _solver(case)
    assert exc.value.code == "FRICTION_CLOSED_TENDON"


def test_preflight_reads_the_profile_back_as_a_stress_at_the_far_end():
    """`prestress` means the JACKING stress once there is a friction block, so
    preflight says which reading is in force and gives the number an analyst
    checks: what is left at the far end, and the percentage lost."""
    from vfife.cli import build
    from vfife.io import schema
    from vfife.preflight import check

    lowered = InputCase(**_case(friction={"mu": MU, "k": K, "jack": "start",
                                          "anchor_set": 0.006})).lower()
    c = schema.Case(**lowered)
    mesh, _material, element = build(c, "unused.yaml")
    issues = {i["code"]: i for i in check(c, mesh, element)["issues"]}

    p = issues["PRESTRESS_PROFILE"]
    assert "AT THE JACK" in p["message"] and "anchor set" in p["message"]
    theta, length = p["details"]["angle"], p["details"]["length"]
    assert p["details"]["far"] == pytest.approx(
        SIG_CON * math.exp(-(MU * theta + K * length)), rel=1e-12)
    # and the losses line now names what the solver DOES compute
    assert "friction, anchor set and elastic shortening" in \
        issues["PRESTRESS_LOSSES"]["message"]


def test_preflight_still_disclaims_friction_when_there_is_no_friction_block():
    from vfife.cli import build
    from vfife.io import schema
    from vfife.preflight import check

    lowered = InputCase(**_case()).lower()
    c = schema.Case(**lowered)
    mesh, _material, element = build(c, "unused.yaml")
    issues = {i["code"]: i for i in check(c, mesh, element)["issues"]}
    assert "not computed here" in issues["PRESTRESS_PROFILE"]["message"]
    assert "friction, anchor set, relaxation" in issues["PRESTRESS_LOSSES"]["message"]
