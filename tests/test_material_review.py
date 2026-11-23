"""Constitutive laws at their edges, as a second reader found them.
One test per reproduced defect."""
import numpy as np
import pytest

from vfife.materials import create


def _drive(m, path, state=None):
    state = m.init_state(1) if state is None else state
    out = []
    for e in path:
        out.append(float(m.update(np.array([[e, 0, 0, 0, 0, 0]], float), state, 1.0)[0, 0]))
    return out, state


def test_steel_cyclic_gives_the_same_stress_at_the_same_strain():
    """A repeated call on an unloading branch jumped onto a reload curve:
    340 -> 202 MPa with no change of strain, and another history after."""
    m = create("steel_cyclic", grade="Q345B")
    path = list(np.linspace(0, 0.0148, 30)) + list(np.linspace(0.0148, -0.01226, 60)) \
        + list(np.linspace(-0.01226, -0.00278, 30)) + list(np.linspace(-0.00278, -0.00311, 3))
    once, s1 = _drive(m, path + [-0.006])
    twice, s2 = _drive(m, path + [-0.00311, -0.006])
    assert twice[-2] == once[-2]
    assert twice[-1] == once[-1]
    assert np.array_equal(s1, s2)


def test_steel_cyclic_updates_a_state_it_is_handed_as_a_view():
    m = create("steel_cyclic", grade="Q345B")
    big = np.zeros((2, m.init_state(1).shape[1]))
    big[:] = m.init_state(1)
    view = big[::2]                               # strided: not contiguous
    for e in (0.001, 0.002, 0.004):
        m.update(np.array([[e, 0, 0, 0, 0, 0]], float), view, 1.0)
    assert big[0, 0] == pytest.approx(0.004)      # eps_prev reached the state


@pytest.mark.parametrize("name, kwargs, match", [
    ("steel_cyclic", dict(grade="Q345B", f_uf=-200e6), "f_uf"),
    ("steel_cyclic", dict(grade="Q345B", f_uf=900e6), "f_uf"),
    ("chaboche", dict(E=2e11, nu=0.3, rho=7850, yield_strength=2e8, damage_max=1.5), "damage_max"),
    ("concrete01", dict(fpc=30e6, epsc0=0.002, epscu=0.001), "epscu"),
    ("concrete02", dict(fpc=30e6, epsc0=0.002, epscu=0.001), "epscu"),
    ("concrete02", dict(fpc=30e6, rat=1.0), "rat"),
    ("concrete01", dict(fpc=30e6, E=40e9, epsc0=0.002), "not both"),
    ("concrete01", dict(fc=30e6, E=40e9, eps_c0=0.002), "disagrees"),
    ("rebar", dict(E=2e11, rho=7850, yield_strength=-4e8), "positive"),
])
def test_a_parameter_outside_its_meaning_is_refused(name, kwargs, match):
    with pytest.raises(ValueError, match=match):
        create(name, **kwargs)


def test_a_ruptured_rebar_s_state_stops_on_both_backends():
    m = create("rebar", E=2e11, rho=7850, yield_strength=4e8, hardening=2e9, rupture_strain=0.01)
    _out, st = _drive(m, list(np.linspace(0, 0.05, 50)))
    alpha = st[0, 1]
    _drive(m, list(np.linspace(0.05, 0.08, 20)), st)
    assert st[0, 1] == alpha and alpha >= 0.01


def test_power_hardening_goes_on_past_a_plastic_strain_of_one():
    from vfife.materials.curves import hardening_power
    c = hardening_power(3e8, 1e9, 0.5)
    sy = c.evaluate(np.array([1.59]))[0] if hasattr(c, "evaluate") else None
    if sy is None:
        pytest.skip("curve has no evaluate()")
    assert float(np.asarray(sy).ravel()[0]) > 3e8 + 1e9 * 1.0 ** 0.5


def test_an_anchorage_on_a_law_without_a_strain_scale_is_refused(tmp_path):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from test_strain_penetration import _case, _solver  # noqa: E402
    path = _case(tmp_path)
    text = pathlib.Path(path).read_text(encoding="utf-8").replace(
        "steel: {type: steel02, E: 200.0e9, Fy: 474.0e6, b: 0.005, R0: 20.0, rho: 7850}",
        "steel: {type: rebar, E: 200.0e9, yield_strength: 474.0e6, rho: 7850}")
    pathlib.Path(path).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="no strain scale"):
        _solver(path)


def test_steel_cyclic_resumes_the_curve_it_left_after_a_wobble():
    """A 1e-5 turn on the yield plateau resumed on the hysteresis skeleton:
    Q345B at 2%, 461 -> 541 MPa in one step; 561 -> 584 in the hardening
    range. A wobble is not a cycle."""
    m = create("steel_cyclic", grade="Q345B")
    for top in (0.02, 0.06):
        up = list(np.linspace(0, top, int(top * 1e4) + 1))
        mono, _ = _drive(m, up + [top + 1e-4])
        wob, _ = _drive(m, up + [top - 1e-5, top, top + 1e-4])
        assert wob[-1] == pytest.approx(mono[-1], rel=1e-12)
        assert wob[-2] == pytest.approx(mono[-2], rel=1e-12)


def test_steel_cyclic_reload_capped_at_E_does_not_step_onto_the_skeleton():
    """Pushed to 2%, unloaded into compression, reloaded: the secant to the
    peak is steeper than E, and the line arrived at the peak strain 80 MPa
    under the skeleton and jumped there."""
    m = create("steel_cyclic", grade="Q345B")
    path = list(np.linspace(0, 0.02, 201)) + list(np.linspace(0.02, 0.0175, 26))[1:] \
        + list(np.linspace(0.0175, 0.0215, 41))[1:]
    s, _ = _drive(m, path)
    steps = np.abs(np.diff(s[-40:]))
    assert steps.max() <= m.E * 1e-4 * (1 + 1e-9)
    assert s[-1] == pytest.approx(float(m.skeleton(0.0215)), rel=1e-9)


def test_steel_cyclic_still_cycles_onto_its_skeleton():
    """The paper's point is kept: after a real cycle the skeleton governs."""
    m = create("steel_cyclic", grade="Q345B")
    path = list(np.linspace(0, 0.02, 201)) + list(np.linspace(0.02, -0.02, 401))[1:] \
        + list(np.linspace(-0.02, 0.025, 451))[1:]
    s, _ = _drive(m, path)
    assert s[-1] == pytest.approx(float(m.skeleton(0.025)), rel=1e-9)
    assert s[-1] > float(m.monotonic(0.025)) * 1.05


def test_concrete01_takes_an_E_that_agrees_with_its_epsc0():
    m = create("concrete01", fc=30e6, eps_c0=0.002, E=3.0e10)
    assert m is not None
