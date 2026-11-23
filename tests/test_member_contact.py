"""Contact between line members (1.11): each member a capsule
about its axis.

* The closest points of two segments, the parallel case at the middle of
  the overlap.
* The forces of a set of overlapping capsules, with friction and spins,
  are a null system: they add to zero and so do their moments (the
  friction's couples included) -- what makes a collision conserve momentum
  and angular momentum.
* A beam dropped across another: it touches when free fall has closed the
  gap between the two SURFACES, t = sqrt(2 (h - r1 - r2) / g); it comes to
  rest with the axes r1 + r2 apart, and the supports of the lower beam
  carry its weight.
* Two free beams thrown at each other: momentum is conserved through the
  collision, to round-off.
* A beam sliding along two rails stops in v^2 / (2 mu g).
* A girder standing closer to its support than the radii allow is let be:
  the dead-load state is the one without contact. When its bearing is
  taken away it lands from where it stood, not from a radius above.
* A run stopped in the middle of a contact resumes to the same bits: the
  allowances are in the checkpoint.
* What the contact cannot mean is refused by name.
"""
import numpy as np
import pytest

numba = pytest.importorskip("numba")

from vfife.api import run_case                                   # noqa: E402
from vfife.core import member_contact as mc                      # noqa: E402

G = 9.80665
RHO, SIDE = 7850.0, 0.2


def test_the_closest_points_of_two_segments():
    c = lambda *a: mc._closest(*(np.asarray(x, float) for x in a))
    # crossing at right angles, one above the other
    s, t = c([-1, 0, 0], [1, 0, 0], [0.5, -2, 1], [0.5, 2, 1])
    assert (s, t) == pytest.approx((0.75, 0.5))
    # end to end
    s, t = c([0, 0, 0], [1, 0, 0], [3, 1, 0], [5, 1, 0])
    assert (s, t) == pytest.approx((1.0, 0.0))
    # parallel, overlapping from x = 1 to 2: the middle of the overlap, on both
    s, t = c([0, 0, 0], [2, 0, 0], [1, 0, 1], [4, 0, 1])
    assert s == pytest.approx(0.75) and t == pytest.approx(0.5 / 3.0)
    # an end-on point contact
    s, t = c([0, 0, 0], [2, 0, 0], [1, 0, 0.5], [1, 0, 3])
    assert (s, t) == pytest.approx((0.5, 0.0))


def test_the_contact_forces_are_a_null_system():
    rng = np.random.default_rng(4)
    m = 30
    x = rng.random((2 * m, 3)) * 2.0
    conn = np.arange(2 * m, dtype=np.int64).reshape(m, 2)
    rad = 0.1 + 0.2 * rng.random(m)
    v = rng.standard_normal((2 * m, 3)); w = rng.standard_normal((2 * m, 3))
    n, pi, pj, ps, pt, po = mc._find(x, conn, rad, 4096)
    assert 20 < n <= 4096
    f = np.zeros_like(x); mom = np.zeros_like(x); pen = np.zeros(len(x))
    mass = 1.0 + rng.random(2 * m)
    inertia = 0.01 + 0.01 * rng.random(2 * m)
    inertia[conn[0]] = 0.0                               # one member a bar: no couple on it, still a null system
    active, deepest, _ = mc._forces(n, pi, pj, ps, pt, x, v, w, conn, rad, np.arange(m, dtype=np.int64), m, mass,
                                    inertia, 1e-3, 0.0, 0.5, 0.3, True, np.zeros(0, np.int64), np.zeros(0), 0.05,
                                    f, mom, pen)
    assert np.abs(mom[conn[0]]).max() == 0.0 and np.abs(f[conn[0]]).max() > 0.0
    assert active == n and deepest == pytest.approx(po[:n].max())
    scale = np.abs(f).max()
    assert scale > 0.0 and np.abs(mom).max() > 0.0
    assert np.abs(f.sum(axis=0)).max() < 1e-12 * scale * len(x)
    total = np.cross(x, f).sum(axis=0) + mom.sum(axis=0)
    assert np.abs(total).max() < 1e-11 * scale * len(x)
    # every pair found is a pair that overlaps, and none is missed (against the plain double loop)
    brute = 0
    for i in range(m):
        for j in range(i + 1, m):
            s, t = mc._closest(x[conn[i, 0]], x[conn[i, 1]], x[conn[j, 0]], x[conn[j, 1]])
            d = np.linalg.norm(((1 - t) * x[conn[j, 0]] + t * x[conn[j, 1]])
                               - ((1 - s) * x[conn[i, 0]] + s * x[conn[i, 1]]))
            brute += d < rad[i] + rad[j]
    assert brute == n


def _two_beams(tmp_path, name, *, h=0.6, r_up=0.1, r_lo=0.15, damping=0.5, friction=0.0, total=1.2,
               struct_damping=None, hold_lower=True, E=2.0e9):
    pts = {"A": [-2.0, 0, 0], "B": [2.0, 0, 0],                 # the lower beam, along x
           "C": [0, -1.5, h], "M": [0, 0, h], "D": [0, 1.5, h]}  # the upper one, across it, free
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts,
                  "lines": {"lower": {"through": ["A", "B"], "size": 0.5},
                            "upper": {"through": ["C", "M", "D"], "size": 0.5}}},
         "materials": {"s": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": RHO}},
         "sections": {"b": {"type": "beam", "material": "s", "shape": "rect", "width": SIDE, "height": SIDE}},
         "assign": [{"region": "all", "section": "b", "orientation": [0, 0, 1]}],
         "constraints": [],
         "loads": [{"region": "upper", "gravity": [0, 0, -G], "amplitude": "hold"}],
         "contact": {"members": {"bodies": [{"region": "upper", "radius": r_up},
                                            {"region": "lower", "radius": r_lo}],
                                 "damping": damping, "friction": friction}},
         "solver": {"mode": "dynamic", "backend": "numba", "total_time": total,
                    "damping": struct_damping or {"type": "none"}},
         "output": {"path": str(tmp_path / f"{name}.h5"), "every": 20,
                    "fields": ["displacement", "contact_force", "contact_penetration"],
                    "monitors": {"zc": {"kind": "displacement", "region": "M", "dof": "z"},
                                 "RA": {"kind": "reaction", "region": "A", "dof": "z"},
                                 "RB": {"kind": "reaction", "region": "B", "dof": "z"}}}}
    if hold_lower:
        d["constraints"] = [{"region": "A", "dofs": [1] * 6}, {"region": "B", "dofs": [1] * 6}]
    return d


def _go(d, **kw):
    box = {}
    out = run_case(d, on_ready=lambda s, p: box.setdefault("s", s), **kw)
    return out, box["s"]


def test_a_dropped_beam_touches_when_the_surfaces_meet_and_comes_to_rest_on_the_other(tmp_path):
    h, r1, r2 = 0.6, 0.1, 0.15
    d = _two_beams(tmp_path, "drop", h=h, r_up=r1, r_lo=r2, total=2.0,
                   struct_damping={"type": "viscous", "coeff": 8.0})
    # the viscous term acts on the fall too: release it only once they have met
    d["solver"]["damping"] = {"type": "none"}
    d["output"]["every"] = 5
    out, s = _go(d)
    assert out.status == "COMPLETED_DYNAMIC", out.message
    import h5py
    with h5py.File(tmp_path / "drop.h5") as f:
        t = np.asarray(f["monitors/time"]); z = np.asarray(f["monitors/zc"])
    gap = h - r1 - r2
    t_hit = np.sqrt(2 * gap / G)
    before = t < 0.98 * t_hit
    assert np.allclose(z[before], -0.5 * G * t[before] ** 2, atol=2e-5)             # free fall until then
    # and not after: the lower beam is in the way (it gives -- a soft beam four metres
    # long -- but the fall is no longer free, and the two never pass through each other)
    k = np.searchsorted(t, t_hit * 1.05)
    assert z[k] > -0.5 * G * t[k] ** 2 + 1e-4
    assert z.min() > -(gap + 0.12) and z[-1] > z.min() + 0.02
    assert s.member_contact.initial_pairs == 0


def test_at_rest_the_axes_are_the_two_radii_apart_and_the_supports_carry_the_weight(tmp_path):
    h, r1, r2 = 0.3, 0.1, 0.15
    d = _two_beams(tmp_path, "rest", h=h, r_up=r1, r_lo=r2, total=3.0, damping=1.0,
                   struct_damping={"type": "viscous", "coeff": 6.0})
    out, s = _go(d)
    assert out.status == "COMPLETED_DYNAMIC", out.message
    W = RHO * SIDE * SIDE * 3.0 * G
    R = float(np.asarray(out.monitors["RA"])[-1] + np.asarray(out.monitors["RB"])[-1])
    assert R == pytest.approx(W, rel=2e-2)
    q = s.q.reshape(-1, 6)
    x = s.mesh.nodes + q[:, :3]
    up = s.mesh.regions["upper"]; lo = s.mesh.regions["lower"]
    mid_up = up[np.argmin(np.abs(s.mesh.nodes[up][:, 1]))]
    mid_lo = lo[np.argmin(np.abs(s.mesh.nodes[lo][:, 0]))]
    sep = x[mid_up, 2] - x[mid_lo, 2]
    assert sep == pytest.approx(r1 + r2, abs=1e-4)                                   # resting: microns of overlap
    assert 0.0 < s.member_contact.max_penetration < 1e-4
    assert s.member_contact.n_active >= 1
    # the contact force field: what the lower beam is pushed down with
    assert np.isclose(s.member_contact.nodal_force[lo][:, 2].sum(), -W, rtol=2e-2)


def test_a_collision_of_two_free_beams_conserves_momentum(tmp_path):
    d = _two_beams(tmp_path, "free", h=0.6, hold_lower=False, total=0.6, damping=0.3, friction=0.4)
    d["loads"] = [{"region": "upper", "gravity": [0, 0, -G], "amplitude": "hold"},
                  {"region": "lower", "gravity": [0.3 * G, 0, 0], "amplitude": "hold"}]
    P = []

    def frame(s):
        v = s.v.reshape(-1, 6)[:, :3]; m = s.physical_mass.reshape(-1, 6)[:, :3]
        P.append(((s.step - 0.5) * s.dt, (m * v).sum(axis=0)))       # leapfrog: v is half a step behind
    box = {}
    out = run_case(d, on_ready=lambda s, p: box.setdefault("s", s),
                   on_progress=lambda *a: frame(box["s"]), progress_every=50)
    assert out.status == "COMPLETED_DYNAMIC", out.message
    s = box["s"]
    m_up = RHO * SIDE * SIDE * 3.0; m_lo = RHO * SIDE * SIDE * 4.0
    t = np.array([p[0] for p in P]); mom = np.array([p[1] for p in P])
    # gravity on the upper beam (down) and the push on the lower (along x) are the only
    # external forces: the momentum is theirs times the time, contact or not
    assert np.allclose(mom[:, 2], -m_up * G * t, rtol=1e-6, atol=1e-6 * m_up)
    assert np.allclose(mom[:, 0], m_lo * 0.3 * G * t, rtol=1e-6, atol=1e-6 * m_lo)
    assert np.abs(s.member_contact.nodal_force).max() >= 0.0 and t[-1] > np.sqrt(2 * 0.35 / G)
    # and they did meet: the lower beam was pushed down
    q = s.q.reshape(-1, 6)
    assert q[s.mesh.regions["lower"], 2].min() < -1e-3


def test_a_beam_sliding_on_two_rails_stops_in_v2_over_2_mu_g(tmp_path):
    mu, r = 0.3, 0.1
    pts = {"A1": [-1.0, -3, 0], "B1": [-1.0, 9, 0], "A2": [1.0, -3, 0], "B2": [1.0, 9, 0],
           "C": [-1.5, 0, 2 * r], "D": [1.5, 0, 2 * r]}
    d = {"version": 1, "name": "slide", "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts,
                  "lines": {"rail1": {"through": ["A1", "B1"], "size": 12.0},
                            "rail2": {"through": ["A2", "B2"], "size": 12.0},
                            "slider": {"through": ["C", "D"], "size": 0.5}},
                  "groups": {"rails": ["rail1", "rail2"]}},
         "materials": {"s": {"type": "linear_elastic", "E": 2.0e9, "nu": 0.3, "rho": RHO}},
         "sections": {"b": {"type": "beam", "material": "s", "shape": "rect", "width": SIDE, "height": SIDE}},
         "assign": [{"region": "all", "section": "b", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "rails", "dofs": [1] * 6},
                         # the slider does not roll or yaw: a block on rails
                         {"region": "slider", "dofs": [1, 0, 0, 1, 1, 1]}],
         "amplitudes": {"kick": [[0, 1.0], [0.05, 1.0], [0.0501, 0.0], [1.0, 0.0]]},
         "loads": [{"region": "slider", "gravity": [0, 0, -G], "amplitude": "hold"},
                   {"region": "slider", "gravity": [0, 2.0 * G, 0], "amplitude": "kick"}],
         "contact": {"members": {"bodies": [{"region": "slider", "radius": r}, {"region": "rails", "radius": r}],
                                 "friction": mu, "damping": 1.0}},
         "solver": {"mode": "dynamic", "backend": "numba", "total_time": 3.0, "damping": {"type": "none"}},
         "output": {"path": str(tmp_path / "slide.h5"), "every": 10, "fields": ["displacement"],
                    "monitors": {"y": {"kind": "displacement", "region": "slider", "dof": "y"}}}}
    out, s = _go(d)
    assert out.status == "COMPLETED_DYNAMIC", out.message
    import h5py
    with h5py.File(tmp_path / "slide.h5") as f:
        t = np.asarray(f["monitors/time"]); y = np.asarray(f["monitors/y"])
    t1 = 0.05 * 3.0                                               # the push: (2 - mu) g for t1
    a1 = (2.0 - mu) * G; v1 = a1 * t1
    stop = 0.5 * a1 * t1 ** 2 + v1 ** 2 / (2 * mu * G)
    assert y[-1] == pytest.approx(stop, rel=1.5e-2)
    assert abs(y[-1] - y[-20]) < 1e-4 * stop                      # and it has stopped
    w = (t > t1 + 0.1) & (t < t1 + v1 / (mu * G) - 0.1)
    assert 2 * np.polyfit(t[w], y[w], 2)[0] == pytest.approx(-mu * G, rel=1e-2)


def _girder_on_cap(tmp_path, name, contact=True, removal=None):
    """A girder 0.05 m above a cap beam on two bearings (connectors): with radii
    of 0.5 and 0.4 the capsules overlap by 0.85 m before anything happens."""
    pts = {"P1": [0, -2, 0], "P2": [0, 2, 0],                    # the cap beam, across
           "G1": [-4, 0, 0.05], "G0": [0, 0, 0.05], "G2": [4, 0, 0.05],
           "K": [0, 0, 0]}                                       # the cap's point under the girder
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts,
                  "lines": {"cap": {"through": ["P1", "K", "P2"], "size": 1.0},
                            "girder": {"through": ["G1", "G0", "G2"], "size": 1.0},
                            "bearing": {"through": ["K", "G0"], "size": 1.0}}},
         "materials": {"s": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2500.0}},
         "sections": {"g": {"type": "beam", "material": "s", "shape": "rect", "width": 1.0, "height": 1.0},
                      "k": {"type": "connector", "stiffness": [2e9, 1e8, 1e8, 1e7, 1e7, 1e7]}},
         "assign": [{"region": "cap", "section": "g", "orientation": [0, 0, 1]},
                    {"region": "girder", "section": "g", "orientation": [0, 0, 1]},
                    {"region": "bearing", "section": "k", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "P1", "dofs": [1] * 6}, {"region": "P2", "dofs": [1] * 6}],
         "loads": [{"region": "girder", "gravity": [0, 0, -G], "amplitude": "hold"}],
         "solver": {"mode": "dynamic", "backend": "numba", "total_time": 0.6,
                    "damping": {"type": "viscous", "coeff": 30.0}},
         "output": {"path": str(tmp_path / f"{name}.h5"), "every": 20, "fields": ["displacement"],
                    "monitors": {"z": {"kind": "displacement", "region": "G0", "dof": "z"}}}}
    if contact:
        d["contact"] = {"members": {"bodies": [{"region": "girder", "radius": 0.5}, {"region": "cap", "radius": 0.4}],
                                    "damping": 1.0}}
    if removal is not None:
        d["removals"] = [{"region": "bearing", "time": removal, "split": False}]
    return d


def test_what_overlaps_at_the_start_is_let_be(tmp_path):
    o0, s0 = _go(_girder_on_cap(tmp_path, "g0", contact=False))
    o1, s1 = _go(_girder_on_cap(tmp_path, "g1", contact=True))
    assert s1.member_contact.initial_pairs >= 1
    # no force from the overlap: the same dead-load history, to the bit
    assert np.array_equal(s0.q, s1.q)
    assert s1.member_contact.n_active == 0 and float(np.abs(s1.member_contact.nodal_force).max()) == 0.0


def test_a_girder_whose_bearing_goes_lands_from_where_it_stood(tmp_path):
    d = _girder_on_cap(tmp_path, "g2", contact=True, removal=0.3)
    d["solver"]["total_time"] = 1.2
    out, s = _go(d)
    assert out.status == "COMPLETED_DYNAMIC", out.message
    z = np.asarray(out.monitors["z"])
    # without the contact it would fall 0.5 g 0.3^2 = 0.44 m in what is left of the run;
    # with it, it drops by the slack, 5% of the smaller radius = 0.02 m, and is held there
    assert z[len(z) // 4 - 2] - z[-1] == pytest.approx(0.02, abs=2e-3)
    assert s.member_contact.n_active >= 1
    W = 2500.0 * 1.0 * 8.0 * G
    assert np.isclose(-s.member_contact.nodal_force[s.mesh.regions["cap"]][:, 2].sum(), W, rtol=5e-2)


def test_a_run_stopped_in_contact_resumes_to_the_same_bits(tmp_path):
    from vfife.core.diagnostics import AnalysisInterrupted
    d = _girder_on_cap(tmp_path, "w", contact=True, removal=0.2)
    d["solver"]["total_time"] = 0.4
    _w, sw = _go(d)
    part = dict(d, name="part", output=dict(d["output"], path=str(tmp_path / "part.h5")))
    n = {"k": 0}

    def stop(*_a):
        n["k"] += 1
        return n["k"] < 6
    with pytest.raises(AnalysisInterrupted):
        _go(part, on_progress=stop, progress_every=max(1, int(sw.n_steps * 0.13)))
    rest = dict(d, name="rest", output=dict(d["output"], path=str(tmp_path / "rest.h5")))
    _r, sr = _go(rest, restart=str(tmp_path / "part.h5"))
    assert sr.step == sw.step
    assert np.array_equal(sr.q, sw.q) and np.array_equal(sr.v, sw.v)
    assert np.array_equal(sr.member_contact.keys[sr.member_contact.vals > 0],
                          sw.member_contact.keys[sw.member_contact.vals > 0])


def test_what_the_contact_cannot_mean_is_refused(tmp_path):
    d = _two_beams(tmp_path, "bad")
    d["contact"]["members"]["bodies"][0].pop("radius")            # a beam with `auto`
    with pytest.raises(ValueError, match="needs `radius`"):
        _go(d)
    d = _two_beams(tmp_path, "bad2")
    d["contact"]["members"]["bodies"].append({"region": "upper", "radius": 0.3})
    with pytest.raises(ValueError, match="one radius"):
        _go(d)
    d = _two_beams(tmp_path, "bad3")
    d["contact"]["members"]["bodies"][0]["radius"] = -1.0
    with pytest.raises(Exception, match="positive"):
        _go(d)


def test_the_neighbour_list_misses_no_pair_as_the_members_move():
    """Between two sweeps a step tests the listed pairs alone; the list is
    made a skin wide and remade when a point has moved half of it. Against
    the full sweep at every step of a random motion."""
    rng = np.random.default_rng(11)
    m = 40
    x = rng.random((2 * m, 3)) * 3.0
    conn = np.arange(2 * m, dtype=np.int64).reshape(m, 2)
    rad = 0.08 + 0.1 * rng.random(m)
    ci = np.zeros(8, np.int64); cj = np.zeros(8, np.int64)
    lstate = np.array([0, 0, 1], np.int64); ref = np.zeros_like(x); stats = np.zeros(4)
    grown = 0
    vel = rng.standard_normal(x.shape) * 0.0008
    hits = 0
    for step in range(600):
        x = x + vel
        before = (lstate.copy(), ref.copy(), stats.copy())
        nc = mc._keep_list(x, conn, rad, ci, cj, lstate, ref, stats)
        if nc < 0:                                    # too short: nothing changed, come back with longer
            assert all(np.array_equal(a, b) for a, b in zip(before, (lstate, ref, stats)))
            ci = np.zeros(-2 * nc, np.int64); cj = np.zeros(-2 * nc, np.int64); grown += 1
            nc = mc._keep_list(x, conn, rad, ci, cj, lstate, ref, stats)
        n, pi, pj, _s, _t, po = mc._narrow(nc, ci, cj, x, conn, rad, max(nc, 1))
        n2, qi, qj, _s2, _t2, qo = mc._find(x, conn, rad, 4096)
        a = sorted(zip(pi[:n].tolist(), pj[:n].tolist())); b = sorted(zip(qi[:n2].tolist(), qj[:n2].tolist()))
        assert a == b, step
        assert a == list(zip(pi[:n].tolist(), pj[:n].tolist()))          # and in the pairs' own order
        hits += n
    assert hits > 500 and 3 < lstate[1] < 150 and grown >= 1            # it met pairs, and swept rarely


# ----------------------------------------------------------- found in review
def test_a_question_is_not_a_step(tmp_path):
    """A probe of another state must not move the allowances: one evaluation
    with the girder lifted cut them, and the force at the state it came
    back to was 4e11 N."""
    from vfife.testing.state import probe
    d = _girder_on_cap(tmp_path, "pr", contact=True)
    d["solver"]["total_time"] = 0.02
    _o, s = _go(d)
    c = s.member_contact
    before = c.vals.copy()
    p = probe(s)
    x = p.get()
    f0 = p.internal(x)
    lifted = x.copy().reshape(-1, 6)
    lifted[s.mesh.regions["girder"], 2] += 0.3
    p.internal(lifted.ravel())
    assert np.array_equal(c.vals, before)
    assert np.array_equal(p.internal(x), f0)


def test_a_fixed_member_stops_what_lands_on_it_whatever_its_density(tmp_path):
    """The contact spring took the reduced mass of the two members' points,
    and a held point's mass is its material's: with a light obstacle (the
    density a member is given when its mass is lumped elsewhere) the spring
    was soft and the falling beam sank 169 mm into it."""
    pen = []
    for rho in (RHO, 0.1):
        d = _two_beams(tmp_path, f"rho{int(rho)}", h=0.3, total=0.6, damping=1.0,
                       struct_damping={"type": "viscous", "coeff": 6.0})
        d["materials"]["light"] = {"type": "linear_elastic", "E": 2.0e9, "nu": 0.3, "rho": rho}
        d["sections"]["o"] = dict(d["sections"]["b"], material="light")
        d["assign"] = [{"region": "upper", "section": "b", "orientation": [0, 0, 1]},
                       {"region": "lower", "section": "o", "orientation": [0, 0, 1]}]
        d["constraints"] = [{"region": "lower", "dofs": [1] * 6}]            # the whole obstacle held
        d["mesh"]["lines"]["lower"]["size"] = 4.0
        worst = {"p": 0.0}
        box = {}
        run_case(d, on_ready=lambda s, p: box.setdefault("s", s), progress_every=20,
                 on_progress=lambda *a: worst.__setitem__("p", max(worst["p"], box["s"].member_contact.max_penetration)))
        pen.append(worst["p"])
    assert pen[0] < 2e-3 and pen[1] < 2e-3
    assert pen[1] == pytest.approx(pen[0], rel=0.2)


def test_a_cable_takes_no_couple_from_friction(tmp_path):
    """A bar's points have no rotary inertia; friction's couple there spun
    them to 1e19 rad/s and threw the chain into the air."""
    pts = {"A": [-1.0, 0, 0], "B": [1.0, 0, 0], "C": [0, -1.0, 0.16], "D": [0, 1.0, 0.16]}
    d = {"version": 1, "name": "chain", "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": pts,
                  "lines": {"beam": {"through": ["A", "B"], "size": 0.5},
                            "chain": {"through": ["C", "D"], "size": 0.25}}},
         "materials": {"s": {"type": "linear_elastic", "E": 2.0e9, "nu": 0.3, "rho": RHO}},
         "sections": {"b": {"type": "beam", "material": "s", "shape": "rect", "width": 0.2, "height": 0.2},
                      "c": {"type": "bar", "material": "s", "shape": "circle", "diameter": 0.1}},
         "assign": [{"region": "beam", "section": "b", "orientation": [0, 0, 1]},
                    {"region": "chain", "section": "c"}],
         "constraints": [{"region": "beam", "dofs": [1] * 6}],
         "loads": [{"region": "chain", "gravity": [0.5 * G, 0, -G], "amplitude": "hold"}],
         "contact": {"members": {"bodies": [{"region": "beam", "radius": 0.1}, {"region": "chain"}],
                                 "friction": 0.2, "damping": 1.0}},
         "solver": {"mode": "dynamic", "backend": "numba", "total_time": 0.3, "damping": {"type": "none"}},
         "output": {"path": str(tmp_path / "chain.h5"), "every": 50, "fields": ["displacement"]}}
    out, s = _go(d)
    assert out.status == "COMPLETED_DYNAMIC", out.message
    assert s.member_contact.radius[s.mesh.element_regions["chain"]["truss2"]] == pytest.approx(0.05)
    v = s.v.reshape(-1, 6)
    assert np.abs(v[:, 3:]).max() < 1e3 and np.abs(v[:, :3]).max() < 20.0
    q = s.q.reshape(-1, 6)
    chain = s.mesh.regions["chain"]
    assert q[chain, 2].max() < 0.05 and q[chain, 2].min() > -1.0          # draped over the beam, not thrown up
    assert np.abs(s.member_contact.nodal_moment[chain]).max() == 0.0


def test_a_member_that_breaks_keeps_its_other_contacts(tmp_path):
    """The two sides of a break get an allowance; the broken member's contact
    with a third member is a contact still (every pair of a member that
    split got one, and a beam resting on another sank a slack into it)."""
    c = mc.MemberContact.__new__(mc.MemberContact)
    c._cap = 64; c.slack = 0.05; c.nel = 3
    c._rad_of = np.array([0.1, 0.1, 0.1]); c.keys = np.zeros(0, np.int64); c.vals = np.zeros(0); c._pending = None
    # members 0 and 1 in line (they shared point 1 before the break; 1 is on point 4 now),
    # member 2 lying across member 1, in contact with it
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [1.5, -1, 0.19], [1.0, 0, 0], [1.5, 1, 0.19]])
    before = np.array([[0, 1], [1, 2], [3, 5]])
    now = np.array([[0, 1], [4, 2], [3, 5]], dtype=np.int64)
    n = c._grant(x, now, c._rad_of.copy(), np.arange(3, dtype=np.int64), shared=before)
    assert n == 1 and c.keys.tolist() == [0 * 3 + 1]                       # the break alone
    assert c.vals[0] == pytest.approx(0.2 + 0.05 * 0.1)


def test_a_member_born_later_is_let_be_where_it_is_born(tmp_path):
    """An analysis step's `activate`: the girder born 0.05 m over its cap got
    no allowance (they are given at the first evaluation), met 0.85 m of
    overlap, and left at 58 km/s."""
    d = _girder_on_cap(tmp_path, "born", contact=True)
    # no bearing here (a connector cannot be born later): the girder is held at its own ends
    del d["mesh"]["lines"]["bearing"], d["sections"]["k"]
    d["assign"] = d["assign"][:2]
    d["constraints"] += [{"region": "G1", "dofs": [1] * 6}, {"region": "G2", "dofs": [1] * 6}]
    d["solver"] = {"mode": "relaxation", "backend": "numba", "n_steps": 40000, "tolerance": 1e-6}
    d["loads"] = [{"region": "girder", "gravity": [0, 0, -G]}]
    d["steps"] = [{"name": "cap", "n_steps": 4000, "deactivate": ["girder"]},
                  {"name": "girder", "activate": ["girder"], "n_steps": 30000},
                  {"name": "shake", "procedure": "dynamic", "total_time": 0.05}]
    out, s = _go(d)
    assert out.status in ("CONVERGED", "COMPLETED_DYNAMIC"), (out.status, out.message)
    q = s.q.reshape(-1, 6); v = s.v.reshape(-1, 6)
    assert np.abs(q[:, :3]).max() < 5e-3 and np.abs(v[:, :3]).max() < 1.0
    assert s.member_contact.n_active == 0


def test_static_refuses_contact_between_members(tmp_path):
    d = _two_beams(tmp_path, "st")
    d["solver"] = {"mode": "static", "backend": "numba", "load_steps": 2}
    with pytest.raises(Exception, match="contact between members"):
        _go(d)
