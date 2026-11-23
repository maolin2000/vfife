"""`path`: a bar route through control points, straight or parabolic.

WHY IT EXISTS
    `line`, `layer` and `hoop` draw straight bars. That is enough for a cage
    and not enough for two things people actually build: a bent-up bar, and a
    prestressing tendon that is draped. Version 0.1 added prestress and had
    to say in Known limits that a curved tendon could not be drawn at all.

THE TWO SHAPES ARE NOT A SMOOTHING PREFERENCE
    `polyline` is exact -- the bar goes through the points and nowhere else --
    and is what a harped tendon, a bent-up bar and a hook are. `parabola` is
    the unique quadratic through three points, which is what a draped tendon
    is, and it is sampled at EQUAL ARC LENGTH because `segment_length` is a
    physical smearing length (BarLineSpec says so) and not a parameter step.

WHAT THE CURVE IS FOR
    A tendon's uplift comes entirely from the angle change between consecutive
    segments: `w = 8 P a / L^2` for a parabola with sag `a`. That is measured
    in cases/case_14, where a downward load of exactly that size leaves the
    beam flat. Here the profile itself is pinned.
"""
import math

import numpy as np
import pytest

from vfife.io.model import InputCase, ReinforcementGeometry


def _nodes(entries):
    """The route a set of line entries describes. One entry, one bar."""
    assert len(entries) == 1, "a path is one line entry: one route, one bar"
    return np.array(entries[0]["points"], dtype=float)


# --------------------------------------------------------------------------- #
#  polyline
# --------------------------------------------------------------------------- #

def test_a_polyline_is_ONE_line_entry_running_through_its_points():
    """One entry, one bar. Two entries whose ends touch are two bars with
    their own nodes -- harmless with perfect bond, and with a `bond` block a
    bar that slips and pulls out in pieces (io/schema.BarLineSpec)."""
    g = ReinforcementGeometry(path={"points": [[0, 0, 0], [1, 0, 0.2], [2, 0, 0]]})
    entries = g.expand("bent", segment_length=0.5)
    assert len(entries) == 1
    assert entries[0]["points"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [2.0, 0.0, 0.0]]
    # each leg is still subdivided by the solver, as a straight line is
    assert entries[0]["segment_length"] == 0.5
    assert "divisions" not in entries[0]


def test_a_polyline_leg_is_never_zero_length():
    """A repeated control point would put a zero-length bar element in the
    model, which `precompute` refuses much further downstream and with much
    less to say about where it came from."""
    g = ReinforcementGeometry(path={"points": [[0, 0, 0], [1, 0, 0], [1, 0, 0]]})
    with pytest.raises(ValueError, match="zero-length leg"):
        g.expand("bent", segment_length=0.5)


def test_the_paths_own_segment_length_wins_over_the_assignments():
    g = ReinforcementGeometry(path={"points": [[0, 0, 0], [2, 0, 0]],
                                    "segment_length": 0.1})
    assert g.expand("p", segment_length=0.5)[0]["segment_length"] == 0.1


# --------------------------------------------------------------------------- #
#  parabola
# --------------------------------------------------------------------------- #

L, SAG = 8.0, 0.20
PROFILE = [[0.0, 0.0, 0.30], [L / 2, 0.0, 0.30 - SAG], [L, 0.0, 0.30]]


def test_a_parabola_passes_through_all_three_of_its_points():
    """Interpolating, not a control polygon: an analyst who writes the low
    point at 0.10 m means the tendon is at 0.10 m there."""
    legs = ReinforcementGeometry(path={"points": PROFILE, "shape": "parabola",
                                       "divisions": 32}).expand("t")
    p = _nodes(legs)
    assert p[0] == pytest.approx(PROFILE[0])
    assert p[-1] == pytest.approx(PROFILE[2])
    assert p[len(p) // 2] == pytest.approx(PROFILE[1], abs=1e-12)


def test_the_chords_are_equal_in_ARC_LENGTH_not_in_parameter():
    """Uniform sampling of t puts longer chords where the curve is steepest --
    2% on this shallow tendon, much more on a sharply draped one -- and
    `segment_length` is a length in metres, so it has to mean one."""
    legs = ReinforcementGeometry(path={"points": PROFILE, "shape": "parabola",
                                       "segment_length": 0.25}).expand("t")
    p = _nodes(legs)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert d.max() / d.min() - 1.0 < 1e-4
    assert d.mean() == pytest.approx(0.25, rel=0.02)


def test_the_number_of_chords_follows_the_arc_length():
    length = ReinforcementGeometry(
        path={"points": PROFILE, "shape": "parabola", "divisions": 400}
    ).expand("t")
    p = _nodes(length)
    arc = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    assert arc > L                                   # a sagging tendon is longer
    entries = ReinforcementGeometry(path={"points": PROFILE, "shape": "parabola",
                                          "segment_length": 0.25}).expand("t")
    assert len(entries[0]["points"]) - 1 == round(arc / 0.25)


def test_the_discretised_profile_turns_through_the_parabolas_own_angle():
    """The uplift of a draped tendon IS this angle change, so it is the
    property the load-balancing case in cases/case_14 depends on. The true
    end-to-end tangent change is 2 atan(4a/L); a chord polygon reaches it from
    below and converges as the sampling refines.
    """
    exact = 2.0 * math.atan(4.0 * SAG / L)
    previous = 0.0
    for n in (8, 32, 128):
        p = _nodes(ReinforcementGeometry(
            path={"points": PROFILE, "shape": "parabola", "divisions": n}).expand("t"))
        d = np.diff(p, axis=0)
        t = d / np.linalg.norm(d, axis=1)[:, None]
        turn = float(np.arccos(np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1, 1)).sum())
        assert turn < exact
        assert turn > previous
        previous = turn
    assert previous == pytest.approx(exact, rel=0.01)


def test_a_parabola_needs_a_sampling_length_from_somewhere():
    """Every other primitive can take `segment_length` from the assignment
    after it has been expanded, because a straight leg is subdivided later.
    Sampling a curve IS the discretisation, so nothing can supply it later."""
    g = ReinforcementGeometry(path={"points": PROFILE, "shape": "parabola"})
    with pytest.raises(ValueError, match="sampling the curve"):
        g.expand("t")
    # ... but the assignment's value does reach it
    assert len(g.expand("t", segment_length=0.25)[0]["points"]) > 2


def test_a_parabola_takes_exactly_three_points():
    with pytest.raises(ValueError, match="exactly three points"):
        ReinforcementGeometry(path={"points": PROFILE + [[9.0, 0.0, 0.3]],
                                    "shape": "parabola"})


def test_repeat_makes_parallel_copies_of_the_whole_route():
    """Four parallel tendons on one profile is one entry and an offset, the
    same way a layer of stirrups is -- and each copy is still one bar."""
    from vfife.elements import truss
    entries = ReinforcementGeometry(
        path={"points": PROFILE, "shape": "parabola", "divisions": 16,
              "repeat": {"count": 4, "offset": [0.0, 0.04, 0.0]}}).expand("t")
    assert len(entries) == 1 and entries[0]["repeat"]["count"] == 4
    nodes, conn = truss.bars_from_spec({"lines": entries})
    assert len(conn) == 64 and len(nodes) == 4 * 17
    joined = sum(1 for i in range(len(conn) - 1) if conn[i][1] == conn[i + 1][0])
    assert joined == 4 * 15, "each copy is one connected bar, and the copies are separate"


# --------------------------------------------------------------------------- #
#  what is refused
# --------------------------------------------------------------------------- #

def test_a_gmsh_curve_as_a_route_is_still_refused_and_says_what_to_do():
    """`path: {group: ...}` was the placeholder this primitive replaced part
    of. Reading a curve out of the mesh file is a different feature and is not
    quietly accepted as if it worked."""
    g = ReinforcementGeometry(path={"group": "tendon"})
    with pytest.raises(ValueError, match="not supported yet"):
        g.expand("t", segment_length=0.25)


def test_points_and_group_are_mutually_exclusive():
    with pytest.raises(ValueError, match="`points` or `group`"):
        ReinforcementGeometry(path={"points": PROFILE, "group": "t"})


def test_a_path_needs_two_points():
    with pytest.raises(ValueError, match="at least two points"):
        ReinforcementGeometry(path={"points": [[0, 0, 0]]})


# --------------------------------------------------------------------------- #
#  end to end
# --------------------------------------------------------------------------- #

def _case(path_spec, prestress=None):
    assign = {"section": "t", "reinforcement": "t", "name": "t"}
    if prestress:
        assign["prestress"] = prestress
    return {
        "mesh": {"dimensions": [8.0, 0.3, 0.6], "divisions": [8, 2, 3]},
        "materials": {"c": {"type": "linear_elastic", "E": 34.0e9, "nu": 0.2,
                            "rho": 2400.0},
                      "s": {"type": "rebar", "E": 195.0e9, "rho": 7850.0,
                            "yield_strength": 1580.0e6}},
        "sections": {"b": {"type": "solid", "material": "c"},
                     "t": {"type": "bar", "material": "s", "shape": "circle",
                           "diameter": 0.014, "area": 120.0e-6}},
        "reinforcement": {"t": {"path": path_spec}},
        "assign": [{"section": "b", "region": "all"}, assign],
        "name": "p",
        "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 10},
        "output": {"path": "unused.h5"},
    }


def test_a_path_reaches_the_solver_as_bars_on_the_curve():
    from vfife.io import schema

    lowered = InputCase(**_case({"points": PROFILE, "shape": "parabola",
                                 "segment_length": 0.25})).lower()
    case = schema.Case(**lowered)
    assert len(case.reinforcement[0].lines) == 1
    assert len(case.reinforcement[0].lines[0].points) == 33

    # and the bar nodes really sit on the parabola, to the solver
    from vfife.elements import truss
    nodes, conn = truss.bars_from_spec(
        {"lines": [ln.model_dump(exclude_none=True) for ln in case.reinforcement[0].lines]})
    assert len(conn) == 32
    x = nodes[:, 0]
    z_exact = 0.30 - SAG * (1.0 - (2.0 * x / L - 1.0) ** 2)
    assert np.allclose(nodes[:, 2], z_exact, atol=1e-9)


def test_preflight_counts_a_path_as_one_tendon_and_not_one_per_chord():
    """THE DEFECT THIS PINS. The force report counts bars, and a bar is a run
    of lines that JOIN -- a parabola sampled into 32 chords with `repeat: 4`
    is four tendons, not 128. Counting entries reported 15360 kN of prestress
    in a member that has 480 kN in it, which is exactly the arithmetic the
    report exists to catch.
    """
    from vfife.cli import build
    from vfife.io import schema
    from vfife.preflight import check

    lowered = InputCase(**_case(
        {"points": PROFILE, "shape": "parabola", "segment_length": 0.25,
         "repeat": {"count": 4, "offset": [0.0, 0.04, 0.0]}},
        prestress=1000.0e6)).lower()
    case = schema.Case(**lowered)
    mesh, _material, element = build(case, "unused.yaml")
    issues = {i["code"]: i for i in check(case, mesh, element)["issues"]}
    assert issues["PRESTRESS"]["details"]["lines"] == 4
    assert issues["PRESTRESS_LOSSES"]["details"]["total"] == pytest.approx(480.0e3)
    # and the profile is reported, because friction is the analyst's
    angle = issues["PRESTRESS_PROFILE"]["details"]["angle"]
    assert angle == pytest.approx(2.0 * math.atan(4.0 * SAG / L), rel=0.05)


# --------------------------------------------------------------------------- #
#  A route is ONE bar
# --------------------------------------------------------------------------- #

def test_a_route_is_laid_as_one_connected_bar():
    """THE DEFECT THIS PINS. `bars_from_spec` gives every line entry its own
    nodes, so entries that merely touch are not joined. A parabolic tendon
    emitted as 32 entries came out as 32 independent two-node stubs -- 64 bar
    nodes for 33 distinct points, zero shared. Perfect bond hid it (coincident
    bar nodes take the same host displacement), a `bond` block would not have.
    """
    from vfife.elements import truss
    entries = ReinforcementGeometry(path={"points": PROFILE, "shape": "parabola",
                                          "segment_length": 0.25}).expand("t")
    nodes, conn = truss.bars_from_spec({"lines": entries})
    assert len(nodes) == 33 and len(conn) == 32
    assert all(conn[i][1] == conn[i + 1][0] for i in range(len(conn) - 1))
    degree = np.bincount(conn.ravel(), minlength=len(nodes))
    assert int((degree == 1).sum()) == 2, "a tendon has two ends, not sixty-four"


def test_a_hoop_is_a_closed_loop_with_no_free_end():
    """And so the refusal in elements/bonded.py -- "a closed hoop has none" --
    is true. Drawn as four touching legs, every tie had eight free ends and
    `anchorage: ends` tied all of them."""
    from vfife.elements import truss
    entries = ReinforcementGeometry(hoop={
        "corners": [[0.05, 0.05], [0.25, 0.25]], "axis": "x",
        "from": 0.1, "to": 0.5, "spacing": 0.2, "divisions": 2}).expand("h")
    assert len(entries) == 1 and entries[0]["closed"]
    nodes, conn = truss.bars_from_spec({"lines": entries})
    degree = np.bincount(conn.ravel(), minlength=len(nodes))
    assert len(conn) == len(nodes)                      # a loop: as many as nodes
    assert int((degree == 1).sum()) == 0


def test_a_straight_route_and_a_straight_line_are_the_same_bar():
    """`points` is not a different formulation, only a different way to say
    where the bar goes: a two-point route and a start/end line have to produce
    identical geometry, and a route with a redundant middle point too."""
    from vfife.elements import truss
    a, b = [0.0, 0.1, 0.1], [2.0, 0.1, 0.1]
    plain = truss.bars_from_spec({"lines": [{"start": a, "end": b, "segment_length": 0.25}]})
    route = truss.bars_from_spec({"lines": [{"points": [a, b], "segment_length": 0.25}]})
    split = truss.bars_from_spec(
        {"lines": [{"points": [a, [1.0, 0.1, 0.1], b], "segment_length": 0.25}]})
    for other in (route, split):
        assert np.allclose(other[0], plain[0])
        assert np.array_equal(other[1], plain[1])


def test_preflight_warns_when_a_bonded_bar_is_drawn_as_touching_pieces():
    """A hand-written bent bar as two `lines` entries is two bars. With perfect
    bond that is harmless; with a `bond` block each piece slips and pulls out
    on its own, so preflight says so and names the fix."""
    from vfife.cli import build
    from vfife.io import schema
    from vfife.preflight import check

    case = _case({"points": PROFILE, "shape": "parabola", "segment_length": 0.5})
    case["assign"][1]["bond"] = {"type": "bond_ceb", "fc": 40.0e6}
    lowered = InputCase(**case).lower()
    # the path itself is one route and must NOT warn
    c = schema.Case(**lowered)
    mesh, _m, element = build(c, "unused.yaml")
    assert "SPLIT_BAR_ROUTE" not in {i["code"] for i in check(c, mesh, element)["issues"]}

    # ... but the same bar written as two touching entries does
    lowered["reinforcement"][0]["lines"] = [
        {"start": PROFILE[0], "end": PROFILE[1], "segment_length": 0.5},
        {"start": PROFILE[1], "end": PROFILE[2], "segment_length": 0.5},
    ]
    c = schema.Case(**lowered)
    issues = {i["code"]: i for i in check(c, mesh, element)["issues"]}
    assert issues["SPLIT_BAR_ROUTE"]["severity"] == "WARNING"
    assert "points" in issues["SPLIT_BAR_ROUTE"]["message"]
