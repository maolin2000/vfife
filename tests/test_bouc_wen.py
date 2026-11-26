"""The Bouc-Wen law (1.22, materials/bouc_wen.py): closed forms of the
one-dimensional law, the plane (Park, Wen & Ang 1986), and a connector that
carries it."""
import math

import numpy as np
import pytest

from vfife.materials import create

E, FY = 2.0e7, 5.0e4          # a bearing's force-deformation, area = length = 1
EY = FY / E


def _drive(m, path):
    st = m.init_state(len(path[:1]))
    out = []
    for e in path:
        s6 = np.zeros((1, 6)); s6[0, 0] = e
        out.append(float(m.update(s6, st, 1.0)[0, 0]))
    return np.asarray(out), st


def test_from_rest_with_n_1_it_is_the_closed_form():
    m = create("bouc_wen", E=E, fy=FY, alpha=0.1, n=1.0)
    eps = np.linspace(0.0, 6 * EY, 301)
    sig, _ = _drive(m, eps)
    z = 1.0 - np.exp(-eps / EY)                       # A = 1, beta + gamma = 1
    assert np.allclose(sig, 0.1 * E * eps + 0.9 * FY * z, rtol=1e-4, atol=1e-5 * FY)
    # one large increment is the same state: the sub-increments
    big, _ = _drive(m, [0.0, 6 * EY])
    assert big[-1] == pytest.approx(sig[-1], rel=1e-3)


def test_unloading_with_beta_equal_gamma_is_elastic_and_a_sharp_knee_is_bilinear():
    m = create("bouc_wen", E=E, fy=FY, n=2.0)
    sig, st = _drive(m, np.linspace(0.0, 4 * EY, 200))
    z = float(st[0, 1])
    s1 = sig[-1]
    s2 = _drive_on(m, st, 4 * EY - 1e-4 * EY)
    # dz/deps on unloading: (A - |z|^n (gamma - beta)) / eps_y = 1 / eps_y
    assert (s1 - s2) / (1e-4 * EY) == pytest.approx(E, rel=1e-3)
    assert 0.0 < z < 1.0
    sharp = create("bouc_wen", E=E, fy=FY, n=30.0)
    sig, _ = _drive(sharp, np.linspace(0.0, 3 * EY, 300))
    assert sig[-1] == pytest.approx(FY, rel=1e-3)
    i = int(np.searchsorted(np.linspace(0.0, 3 * EY, 300), 0.8 * EY))
    assert sig[i] == pytest.approx(E * np.linspace(0.0, 3 * EY, 300)[i], rel=2e-2)


def _drive_on(m, st, e):
    s6 = np.zeros((1, 6)); s6[0, 0] = e
    return float(m.update(s6, st, 1.0)[0, 0])


def test_the_dissipated_energy_is_the_area_of_the_loop():
    m = create("bouc_wen", E=E, fy=FY, alpha=0.05, n=1.5)
    t = np.linspace(0.0, 2.0, 4001)
    eps = 5 * EY * np.sin(2 * np.pi * t)
    sig, st = _drive(m, eps)
    # the second cycle is (nearly) closed: its area is what it dissipated
    half = len(t) // 2
    area = float(np.sum(0.5 * (sig[half + 1:] + sig[half:-1]) * np.diff(eps[half:])))
    _s, st1 = _drive(m, eps[:half + 1])
    assert float(st[0, 2] - st1[0, 2]) == pytest.approx(area, rel=1e-3)
    assert area > 0.0


def test_in_the_plane_a_radial_path_is_the_line_and_the_surface_is_a_circle():
    m = create("bouc_wen", E=E, fy=FY, n=2.0)
    st = m.init_state_planar()
    path = np.linspace(0.0, 4 * EY, 200)
    c, s = math.cos(0.6), math.sin(0.6)
    for e in path:
        f2 = m.update_planar([e * c, e * s], st)
    line, _ = _drive(m, path)
    assert np.hypot(*f2) == pytest.approx(line[-1], rel=1e-6)
    assert f2[1] / f2[0] == pytest.approx(s / c, rel=1e-9)
    # yielded along x, then moved along y: the x force falls, |z| stays within 1
    st = m.init_state_planar()
    for e in np.linspace(0.0, 4 * EY, 200):
        f2 = m.update_planar([e, 0.0], st)
    fx_yielded = f2[0]
    for e in np.linspace(0.0, 4 * EY, 200):
        f2 = m.update_planar([4 * EY, e], st)
        assert math.hypot(st[0, 2], st[0, 3]) <= 1.0 + 1e-9
    assert f2[0] < 0.5 * fx_yielded and f2[1] > 0.8 * FY


def test_refused():
    with pytest.raises(ValueError, match="beta"):
        create("bouc_wen", E=E, fy=FY, beta=0.5, gamma=-0.5)
    with pytest.raises(ValueError, match="alpha"):
        create("bouc_wen", E=E, fy=FY, alpha=1.0)


def test_a_connectors_uy_and_uz_with_one_bouc_wen_law_are_one_law_in_the_plane():
    from vfife.elements import connector
    law = {"type": "material", "material": {"type": "bouc_wen", "E": E, "fy": FY, "n": 2.0}}
    props = {"stiffness": [1e9, 0, 0, 0, 0, 0], "laws": {"uy": law, "uz": law}}
    states = connector.MaterialStates([0], lambda e: props)
    row = states.of(0)
    assert list(row) == [1] and row[1][1].shape == (1, 5)
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R0 = np.eye(3)
    m = create("bouc_wen", E=E, fy=FY, n=2.0)
    st = m.init_state_planar()
    for k in range(1, 101):
        u = np.zeros((2, 3)); u[1] = [0.0, 4 * EY * k / 100, 2 * EY * k / 100]
        _fe, action, d = connector.response(u, None, np.zeros(12), x0, R0, props, materials=row)
        states.commit()
        want = m.update_planar(d[1:3], st)
    assert np.allclose(action[1:3], want, rtol=1e-9)
    # two different laws stay two laws
    other = dict(law, material=dict(law["material"], fy=2 * FY))
    assert not connector.planar_material({"uy": law, "uz": other})


def test_a_bearing_cycled_through_a_model_is_the_law(tmp_path):
    """A connector's uy carrying the law in a staged relaxation, driven
    through a cycle: its force is the law's along the same deformations."""
    import sys
    sys.path.insert(0, "tests")
    from test_connector_material import _cycled
    mat = {"type": "bouc_wen", "E": E, "fy": FY, "alpha": 0.05, "n": 2.0}
    u, F, out, _ = _cycled(tmp_path, "bw", {"type": "material", "material": "bw"}, {"bw": mat}, n=16)
    m = create("bouc_wen", **{k: v for k, v in mat.items() if k != "type"})
    st = m.init_state(1); ref = []
    for x in u:
        ref.append(float(m.update(np.array([[x, 0, 0, 0, 0, 0]]), st, 0.0)[0, 0]))
    assert np.allclose(F, ref, rtol=1e-5, atol=1e-4 * FY)
    assert F.max() > 0.9 * FY


def test_the_energy_is_the_same_in_one_increment_and_the_stiffness_bound_is_the_largest_tangent():
    from vfife.elements import connector
    m = create("bouc_wen", E=E, fy=FY, alpha=0.05, n=1.5)
    path = [0.0, 3 * EY, -3 * EY, 0.0]
    _s, coarse = _drive(m, path)
    _s, fine = _drive(m, np.concatenate([np.linspace(a, b, 400) for a, b in zip(path[:-1], path[1:])]))
    assert float(coarse[0, 2]) == pytest.approx(float(fine[0, 2]), rel=2e-3)
    odd = create("bouc_wen", E=E, fy=FY, beta=0.9, gamma=-0.4, A=2.0)
    assert odd.tangent_max == pytest.approx(E * 2.0 * 1.8 / 0.5)
    law = {"type": "material", "material": {"type": "bouc_wen", "E": E, "fy": FY, "beta": 0.9, "gamma": -0.4}}
    assert connector.law_stiffness(law) == pytest.approx(E * 1.8 / 0.5)
