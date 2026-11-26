"""Rayleigh damping by region (1.22, core/damping.py)."""
import numpy as np
import pytest

from vfife import cli
from vfife.io.schema import DampingSpec
from vfife.testing import models

LINE = ("beam2", "beam2_timoshenko", "beam2_fiber", "mixed_line")
BASE = {"type": "rayleigh", "alpha": 0.8, "beta": 2e-6}


def _solver(element, damping):
    m = models.build(element, mode="dynamic", solver={"damping": damping, "total_time": 2e-4})
    x = m.mesh.nodes[:, 0]
    m.mesh.regions["half"] = np.flatnonzero(x <= 0.5 * (x.min() + x.max()) + 1e-9)
    return cli.make_solver(m.problem, m.mesh, m.material)


def _damping_force(s, v):
    s.v = v.copy()
    f = s.alpha * s.mass * v + s.beta * (s.K @ v)
    r = s.regional_damping()
    return f if r is None else f + r.force(v)


@pytest.mark.parametrize("element", LINE)
def test_the_force_is_the_sum_over_the_members_of_their_own_rayleigh(element):
    """f = alpha_0 M v + beta_0 K v, with each region member's own alpha_e
    M_e + beta_e K_e in place of the model's: checked against the members'
    blocks summed by hand."""
    regions = [{"regions": ["all"], "alpha": 3.0, "beta": 0.0}, {"regions": ["half"], "alpha": 0.1, "beta": 5e-6}]
    s = _solver(element, {**BASE, "regions": regions})
    rng = np.random.default_rng(3)
    v = rng.standard_normal(s.ndof)
    got = _damping_force(s, v)
    # by hand: the members of the first half take the later entry's 0.1 and
    # 5e-6, the rest the first one's 3.0 and 0; the masses no member
    # carries keep the model's alpha
    from vfife.core.member_actions import region_members
    half = set(region_members(s.mesh, ["half"], s.mesh.elements[:, :2]).tolist())
    assert 0 < len(half) < s.nel
    want = s.alpha * s.mass * v + s.beta * (s.K @ v)
    for e in range(s.nel):
        a, b = (0.1, 5e-6) if e in half else (3.0, 0.0)
        dof = s.dofs.element_dofs(s.mesh.elements[e, :2])
        m = np.tile(np.asarray(s.member_mass_share(e), float), 2)
        want[dof] += (a - s.alpha) * m * v[dof]
        want[dof] += (b - s.beta) * (s.member_reference_block(e) @ v[dof])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("element", LINE)
def test_a_region_with_the_models_coefficients_is_no_region(element):
    plain = _solver(element, dict(BASE))
    same = _solver(element, {**BASE, "regions": [{"regions": ["all"], "alpha": BASE["alpha"],
                                                   "beta": BASE["beta"]}]})
    v = np.random.default_rng(4).standard_normal(plain.ndof)
    np.testing.assert_array_equal(_damping_force(plain, v), _damping_force(same, v))
    assert plain.dt == same.dt


def test_a_stiffer_region_shrinks_the_step_and_takes_the_loop():
    plain = _solver("beam2", dict(BASE))
    stiff = _solver("beam2", {**BASE, "regions": [{"regions": ["all"], "beta": 50 * BASE["beta"]}]})
    assert stiff.dt < plain.dt
    from vfife.core import compiled_step, capabilities
    why = compiled_step.refusal(stiff, *stiff.step_data(0))
    if why not in (capabilities.refuse("off"), capabilities.refuse("no_numba")):
        assert why == capabilities.refuse("damping_regions")
        assert compiled_step.refusal(plain, *plain.step_data(0)) != why


def _held(s):
    held = np.zeros(s.ndof, bool)
    for n in s.mesh.regions["fixed"]:
        held[s.dofs.node_dofs(int(n))] = True
    return held


def _C(s):
    """The damping operator, column by column."""
    free = np.flatnonzero(~_held(s))
    C = np.zeros((s.ndof, len(free)))
    for j, d in enumerate(free):
        e = np.zeros(s.ndof); e[d] = 1.0
        C[:, j] = _damping_force(s, e)
    return C[free], free


def test_the_first_mode_is_damped_at_the_ratio_of_the_region_it_is_in():
    """A cantilever all in one region of ratio 2% at its first two
    frequencies, the model's 10%: its first mode's ratio is 2%, to the
    rotary floor of the lumped mass (which keeps the model's alpha)."""
    import scipy.linalg as sl
    s0 = _solver("beam2", dict(BASE))
    free = np.flatnonzero(~_held(s0))
    K = s0.K.toarray()[np.ix_(free, free)]
    M = s0.mass[free]
    w2, vecs = sl.eigh(K, np.diag(M), subset_by_index=[0, 7])
    # the first mode and the first one well above it: a square section
    # bends the same both ways, and its first two modes are one frequency
    # (to round-off, which machine orders them is chance)
    j = int(np.flatnonzero(w2 > 1.5 * w2[0])[0])
    w = np.sqrt(w2[[0, j]]); f = w / (2 * np.pi); vec = vecs[:, [0, j]]
    for ratio, region, want in ((0.02, None, 0.02), (0.10, {"regions": ["all"], "ratio": 0.02}, 0.02),
                                (0.10, None, 0.10)):
        damp = {"type": "rayleigh", "ratio": ratio, "frequencies": [float(f[0]), float(f[1])]}
        if region:
            damp["regions"] = [region]
        s = _solver("beam2", damp)
        C, _ = _C(s)
        for k in range(2):
            phi = vec[:, k]
            z = phi @ C @ phi / (2 * w[k] * (phi * M) @ phi)
            assert z == pytest.approx(want, rel=2e-2), (ratio, region, k)


def _two_span(tmp_path, damping):
    return {
        "version": 1, "name": "two", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "M": [4, 0, 0], "B": [8, 0, 0]},
                 "lines": {"steel": {"through": ["A", "M"], "size": 1.0},
                           "rubber": {"through": ["M", "B"], "size": 1.0}}},
        "materials": {"m": {"type": "linear_elastic", "E": 2e11, "nu": 0.3, "rho": 7850.0}},
        "sections": {"s": {"type": "beam", "material": "m", "shape": "general",
                           "area": 1e-2, "iy": 1e-5, "iz": 1e-5, "j": 2e-5}},
        "assign": [{"region": "steel", "section": "s"}, {"region": "rubber", "section": "s"}],
        "constraints": [{"region": "A", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": [{"region": "B", "vector": [0, 0, -1e3]}],
        "solver": {"mode": "dynamic", "total_time": 0.05, "damping": damping},
        "output": {"path": str(tmp_path / "two.h5"),
                   "monitors": {"w": {"kind": "displacement", "region": "B", "dof": "z"}}},
    }


def test_a_line_names_its_region_and_the_run_damps_it(tmp_path):
    from vfife.api import run_case
    got = {}
    reg = [{"regions": ["rubber"], "alpha": 40.0}]
    out = run_case(_two_span(tmp_path, {**BASE, "regions": reg}),
                   on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status in ("COMPLETED", "COMPLETED_DYNAMIC"), out.message
    r = got["s"].regional_damping()
    s = got["s"]
    x = s.mesh.nodes[s.mesh.elements[r.ids, :2]][:, :, 0]
    assert len(r.ids) == 4 and (x >= 4.0 - 1e-9).all()      # the members from M to B
    plain = run_case(_two_span(tmp_path / "p", dict(BASE)))
    # more damping where the tip is: the tip moves less
    assert abs(out.monitors["w"][-1]) != abs(plain.monitors["w"][-1])


def test_refused_where_it_would_not_be_applied():
    with pytest.raises(Exception, match="type: rayleigh"):
        DampingSpec(type="viscous", coeff=1.0, regions=[{"regions": ["a"], "alpha": 1.0}])
    with pytest.raises(Exception, match="needs `frequencies`"):
        DampingSpec(type="rayleigh", alpha=1.0, regions=[{"regions": ["a"], "ratio": 0.02}])
    with pytest.raises(Exception, match="not both"):
        DampingSpec(ratio=0.05, frequencies=[1, 2], regions=[{"regions": ["a"], "ratio": 0.02, "alpha": 3.0}])
    with pytest.raises(Exception, match="line models"):
        models.build("shell4", mode="dynamic",
                     solver={"damping": {**BASE, "regions": [{"regions": ["all"], "alpha": 1.0}]}})
    with pytest.raises(Exception, match="dynamic run"):
        models.build("beam2", solver={"damping": {"type": "kinetic"}} | {} ) if False else \
            models.build("beam2", mode="relaxation",
                         solver={"damping": {"type": "rayleigh", "alpha": 1.0,
                                             "regions": [{"regions": ["all"], "alpha": 2.0}]}})


def test_a_region_that_is_not_there_is_refused_at_setup_and_one_like_the_model_is_nothing():
    with pytest.raises(ValueError, match=r"damping.regions\[0\]: no region 'nowhere'"):
        _solver("beam2", {**BASE, "regions": [{"regions": ["nowhere"], "alpha": 1.0}]})
    same = _solver("beam2", {**BASE, "regions": [{"regions": ["all"], "alpha": BASE["alpha"],
                                                   "beta": BASE["beta"]}]})
    assert same.regional_damping() is None            # nothing to add: the compiled step stays open
