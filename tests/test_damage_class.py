"""`damage_class`: what state a material point is in, for a picture:
0 intact, 1 cracked, 2 crushed, 3 yielded, 4 failed.

Each law is driven through a strain history and asked; the classes must
follow the events the law itself acts on -- the tensile strength, the
peak compressive strain, the yield surface, the rupture latch.
"""
import numpy as np
import pytest

from vfife.materials.base import create


def _drive(mat, history):
    st = mat.init_state(1)
    for eps in history:
        strain6 = np.zeros((1, 6)); strain6[0, 0] = eps
        mat.update(strain6, st, 1.0)
    return st


def test_concrete02_cracks_past_ft_and_crushes_past_the_peak():
    m = create("concrete02", fpc=-30e6, epsc0=-0.002, fcu=-6e6, epscu=-0.006, rat=0.1, ft=3e6, Ets=1.5e9, rho=2400)
    Ec0 = 2 * 30e6 / 0.002
    assert m.damage_class(_drive(m, [0.0]))[0] == 0
    assert m.damage_class(_drive(m, [0.5 * 3e6 / Ec0]))[0] == 0          # below f_t: intact
    assert m.damage_class(_drive(m, [3.0 * 3e6 / Ec0]))[0] == 1          # past f_t: cracked
    assert m.damage_class(_drive(m, [-0.001]))[0] == 0                   # compression before the peak
    assert m.damage_class(_drive(m, [-0.003]))[0] == 2                   # past epsc0: crushed
    # crushed wins over cracked once both have happened
    assert m.damage_class(_drive(m, [3.0 * 3e6 / Ec0, -0.003]))[0] == 2


def test_concrete01_has_no_tension_so_an_open_crack_is_a_positive_strain():
    m = create("concrete01", fpc=-30e6, epsc0=-0.002, fpcu=-6e6, epscu=-0.006, rho=2400)
    assert m.damage_class(_drive(m, [-0.001]))[0] == 0
    assert m.damage_class(_drive(m, [0.0005]))[0] == 1
    assert m.damage_class(_drive(m, [-0.0025]))[0] == 2


def test_rebar_yields_then_ruptures():
    m = create("rebar", E=200e9, rho=7850, yield_strength=400e6, hardening=2e9, rupture_strain=0.05)
    assert m.damage_class(_drive(m, [0.001]))[0] == 0
    assert m.damage_class(_drive(m, [0.004]))[0] == 3
    assert m.damage_class(_drive(m, [0.004, 0.08]))[0] == 4


def test_steel02_yields_when_a_plastic_residual_appears():
    m = create("steel02", E=200e9, Fy=400e6, b=0.01, rho=7850)
    assert m.damage_class(_drive(m, [0.0005]))[0] == 0
    assert m.damage_class(_drive(m, [0.01]))[0] == 3


def test_an_elastic_law_has_no_classes():
    m = create("linear_elastic", E=30e9, nu=0.2, rho=2400)
    assert m.damage_class(np.zeros((3, 0))) is None
