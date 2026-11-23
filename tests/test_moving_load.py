"""A load that travels along the structure (cases/case_26).

Three claims, each checked against something outside the solver:

  * the consistent distribution of a point load onto a member's ends is the
    Hermite one, and summing it over a member reproduces the `line_load`
    end forces the library already had;
  * so a load standing BETWEEN two nodes gives the exact nodal deflection of
    a prismatic beam -- the whole influence line, not just the stations that
    land on a node; and
  * the path a load travels is one ordered chain of members, and anything
    else is refused by name rather than traversed in mesh order.
"""
import pathlib
import sys

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from vfife.api import _with_element, prepare_case
from vfife.core.moving_load import LoadPath, MovingLoad, consistent_point_load
from vfife.core.structural import make_structural_solver

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))
ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE = "cases/case_26b_moving_load_influence.yaml"
E, I, L, P = 2.1e11, 9.70e-4, 20.0, 1.0e5


def _solver(case=CASE):
    prep = prepare_case(str(ROOT / case))
    return make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                  prep.mesh, prep.material)


def test_the_distribution_is_the_one_the_element_is_built_from():
    """Integrating the point-load distribution over a member must give the
    uniform load's end actions -- qL/2 at each end and +-qL^2/12 -- because
    a uniform load IS point loads side by side. That is the check that these
    signs agree with the `line_load` block rather than merely looking right.
    """
    R = np.eye(3); Lm = 3.0
    for axis, (f_i, m_i, sign) in {1: (1, 5, +1.0), 2: (2, 4, -1.0)}.items():
        p = np.zeros(3); p[axis] = 1.0
        xi = (np.arange(4000) + 0.5) / 4000                 # midpoint rule
        tot = sum(consistent_point_load(Lm, R, x, p / 4000 * Lm) for x in xi)
        assert tot[f_i] == pytest.approx(Lm / 2, rel=1e-6)
        assert tot[f_i + 6] == pytest.approx(Lm / 2, rel=1e-6)
        assert tot[m_i] == pytest.approx(sign * Lm**2 / 12, rel=1e-4)
        assert tot[m_i + 6] == pytest.approx(-sign * Lm**2 / 12, rel=1e-4)
    # axial is the lever rule, and every distribution carries the whole force
    f = consistent_point_load(2.0, R, 0.25, np.array([100.0, 7.0, -3.0]))
    assert f[0] == pytest.approx(75.0) and f[6] == pytest.approx(25.0)
    assert f[1] + f[7] == pytest.approx(7.0) and f[2] + f[8] == pytest.approx(-3.0)
    # and at a node it is simply a nodal force, with no end moments at all
    f = consistent_point_load(2.0, R, 0.0, np.array([0.0, 0.0, -5.0]))
    assert f[2] == pytest.approx(-5.0) and np.allclose(f[[4, 8, 10]], 0.0)


def test_a_load_between_nodes_gives_the_exact_deflection():
    """The influence line for midspan deflection, station by station, against
    P a (3 L^2 - 4 a^2) / 48 EI. Half the stations are in the middle of a
    member; the consistent distribution is exact at the nodes there too,
    which is the whole reason it is used.

    The lever rule -- split the force between the ends by 1 - xi and xi, no
    end moments -- is the tempting cheap alternative, and the last line
    measures what it costs on THIS mesh (0.4%). What it costs on a coarse one
    is the next test.
    """
    s = _solver()
    fixed = s.step_data(0)[0]
    free = ~fixed
    K = s.K.tocsr()[free][:, free]
    mid = s.dofs.node_dofs(s.mesh.regions["mid"], ("uz",)).ravel()[0]
    where = np.flatnonzero(free) == mid
    path = LoadPath(s.mesh, np.arange(s.nel))

    def deflection(station, lever=False):
        f = np.zeros(s.ndof)
        e, xi = path.locate(station)
        if lever:
            conn = s.mesh.elements[e][:2]
            dof = s.dofs.node_dofs(conn, ("uz",)).ravel()
            f[dof[0]] -= P * (1 - xi); f[dof[1]] -= P * xi
        else:
            MovingLoad({"axles": [[0.0, P]], "start": station, "stop": station + 1.0},
                       path, s.dofs, s.mesh, s.length, s.R0, 1.0)(0.0, None, f)
        return -spsolve(K, f[free])[where][0]

    def exact(a):
        a = min(a, L - a)
        return P * a * (3 * L * L - 4 * a * a) / (48 * E * I)

    for a in np.arange(0.5, 19.6, 0.5):
        assert deflection(a) == pytest.approx(exact(a), rel=1e-9), f"station {a}"
    # ... and what the cheap distribution costs here: small, because 1 m
    # members are twenty to a span. It is small, not zero, and it does not
    # go away with refinement in the way an error that IS zero does.
    worst = min(deflection(a, lever=True) / exact(a) for a in np.arange(0.5, 19.6, 0.5))
    assert worst == pytest.approx(0.9964, abs=5e-4)


def test_the_lever_rule_is_a_different_structure_on_a_coarse_mesh():
    """The same two distributions on a beam assembled here, away from the
    solver entirely, so the comparison is arithmetic and not a second opinion
    from the same code.

    With the load in the middle of a member the lever rule hands the whole
    force to the two nodes and NO end moments -- and on one element per half
    span that is not a 1% correction, it is 27% of the answer. The consistent
    distribution is exact at every refinement, including the two-element mesh
    where the load acts nowhere near a degree of freedom.
    """
    def midspan(nel, a, lever):
        le = L / nel; n = nel + 1
        K = np.zeros((2*n, 2*n))
        ke = E*I/le**3*np.array([[12, 6*le, -12, 6*le], [6*le, 4*le*le, -6*le, 2*le*le],
                                 [-12, -6*le, 12, -6*le], [6*le, 2*le*le, -6*le, 4*le*le]])
        for e in range(nel):
            d = [2*e, 2*e+1, 2*e+2, 2*e+3]; K[np.ix_(d, d)] += ke
        f = np.zeros(2*n); e = min(int(a/le), nel-1); xi = (a - e*le)/le; b = 1 - xi
        d = [2*e, 2*e+1, 2*e+2, 2*e+3]
        if lever:
            f[d[0]] -= P*b; f[d[2]] -= P*xi
        else:
            f[d[0]] -= P*(1 - 3*xi**2 + 2*xi**3); f[d[2]] -= P*(3*xi**2 - 2*xi**3)
            f[d[1]] -= P*le*xi*b*b;               f[d[3]] += P*le*xi*xi*b
        free = np.setdiff1d(np.arange(2*n), [0, 2*n - 2])
        u = np.zeros(2*n); u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
        return -u[2*(nel // 2)]

    def exact(a):
        a = min(a, L - a)
        return P*a*(3*L*L - 4*a*a)/(48*E*I)

    for nel in (2, 4, 20):
        for a in (5.0, 7.5, L/2):
            assert midspan(nel, a, False) == pytest.approx(exact(a), rel=1e-9)
    # one element per half span, the load halfway along it
    assert midspan(2, 5.0, True) / exact(5.0) == pytest.approx(0.727, abs=0.002)


def test_axles_ride_on_and_off_and_add_up():
    """A group of axles is superposition on a linear beam, and an axle whose
    station is off the path carries nothing -- which is how a vehicle drives
    on at one end and off at the other rather than being teleported whole."""
    s = _solver()
    path = LoadPath(s.mesh, np.arange(s.nel))
    two = MovingLoad({"axles": [[0.0, P], [-4.0, P]], "start": 0.0, "stop": L},
                     path, s.dofs, s.mesh, s.length, s.R0, 1.0)
    one = MovingLoad({"axles": [[0.0, P]], "start": 0.0, "stop": L},
                     path, s.dofs, s.mesh, s.length, s.R0, 1.0)
    # lead axle at 10 m, trailing axle at 6 m: the sum of the two separately
    f2 = two(0.5, None, np.zeros(s.ndof))
    a = one.__class__({"axles": [[0.0, P]], "start": 10.0, "stop": 10.0 + 1},
                      path, s.dofs, s.mesh, s.length, s.R0, 1.0)(0.0, None, np.zeros(s.ndof))
    b = one.__class__({"axles": [[0.0, P]], "start": 6.0, "stop": 7.0},
                      path, s.dofs, s.mesh, s.length, s.R0, 1.0)(0.0, None, np.zeros(s.ndof))
    assert np.allclose(f2, a + b)
    # at the start only the lead axle is on: the trailing one is at -4 m
    f0 = two(0.0, None, np.zeros(s.ndof))
    assert f0.sum() == pytest.approx(-P)           # one axle's worth of force
    # and the whole group carries its whole weight once it is all on
    assert two(0.5, None, np.zeros(s.ndof)).sum() == pytest.approx(-2 * P)


def test_the_station_is_time_when_it_has_a_speed_and_progress_when_it_has_a_stop():
    s = _solver()
    path = LoadPath(s.mesh, np.arange(s.nel))
    kw = dict(path=path, dofs=s.dofs, mesh=s.mesh, length=s.length, R0=s.R0, total_time=0.5)
    fast = MovingLoad({"axles": [[0.0, P]], "start": -2.0, "speed": 40.0}, **kw)
    assert fast.station(0.0, 0.0) == pytest.approx(-2.0)
    assert fast.station(0.5, 0.25) == pytest.approx(8.0)
    # no clock (a quasi-static probe, or an invariant check): the step's own
    assert fast.station(1.0, None) == pytest.approx(fast.station(1.0, 0.5))
    sweep = MovingLoad({"axles": [[0.0, P]], "start": 1.0, "stop": 19.0}, **kw)
    assert sweep.station(0.0, None) == pytest.approx(1.0)
    assert sweep.station(0.5, 123.0) == pytest.approx(10.0)      # the clock is ignored


def test_the_path_is_one_ordered_chain_or_it_is_refused():
    s = _solver()
    path = LoadPath(s.mesh, np.arange(s.nel))
    assert path.total == pytest.approx(L)
    assert s.mesh.nodes[path.nodes[0]][0] == pytest.approx(0.0)   # s = 0 at the -x end
    assert np.all(np.diff(s.mesh.nodes[path.nodes][:, 0]) > 0)    # ordered, not mesh order
    assert path.locate(-0.1) is None and path.locate(L + 0.1) is None
    e, xi = path.locate(3.25)
    assert s.mesh.nodes[s.mesh.elements[e][0]][0] == pytest.approx(3.0)
    assert xi == pytest.approx(0.25)

    class FakeMesh:
        def __init__(self, nodes, elements):
            self.nodes, self.elements = np.asarray(nodes, float), np.asarray(elements, int)

    fork = FakeMesh([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], [[0, 1], [1, 2], [1, 3]])
    with pytest.raises(ValueError, match="branches"):
        LoadPath(fork, [0, 1, 2])
    ring = FakeMesh([[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 1], [1, 2], [2, 0]])
    with pytest.raises(ValueError, match="one open chain"):
        LoadPath(ring, [0, 1, 2])
    apart = FakeMesh([[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]], [[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="one open chain"):
        LoadPath(apart, [0, 1])
    with pytest.raises(ValueError, match="no complete members"):
        LoadPath(apart, [])


def test_the_case_file_must_say_how_the_load_moves():
    from vfife.io import schema
    base = {"region": "deck", "moving_load": {"axles": [[0.0, 1.0e5]]}}
    with pytest.raises(ValueError, match="exactly one of speed"):
        schema.LoadSpec(**base)
    with pytest.raises(ValueError, match="exactly one of speed"):
        schema.LoadSpec(region="deck", moving_load={"axles": [[0.0, 1e5]], "speed": 10.0, "stop": 5.0})
    with pytest.raises(ValueError, match="never moves"):
        schema.LoadSpec(region="deck", moving_load={"axles": [[0.0, 1e5]], "speed": 0.0})
    with pytest.raises(ValueError, match="offset along the path"):
        schema.LoadSpec(region="deck", moving_load={"axles": [[0.0, 1e5, 3.0]], "speed": 1.0})
    with pytest.raises(ValueError, match="exactly one of vector"):
        schema.LoadSpec(region="deck", vector=[0, 0, -1],
                        moving_load={"axles": [[0.0, 1e5]], "speed": 1.0})
    ok = schema.LoadSpec(region="deck", moving_load={"axles": [[0.0, 1e5]], "speed": 10.0})
    assert ok.moving_load.direction == [0.0, 0.0, -1.0]      # down, unless said otherwise
