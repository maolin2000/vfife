"""Hardening/softening curve construction.

The GB 50010 tests are deliberately property-based rather than golden-value:
they check what the code guarantees (peak stress, peak strain, initial tangent,
monotone softening), so a transcription error in the shape of either curve --
the formulas are reproduced in the module docstring for checking against the
standard -- fails here rather than silently changing every concrete result.
"""

import numpy as np
import pytest

from vfife.materials import curves as cv

E0 = 3.0e10
FC, FT = 30.0e6, 3.0e6


def gb_compression_raw(fc, E0):
    """The GB 50010 C.2.4 curve itself, before any CDP conversion."""
    fc_M, E_M = fc / 1e6, E0 / 1e6
    eps_cr = (700.0 + 172.0 * np.sqrt(fc_M)) * 1e-6
    alpha_c = 0.157 * fc_M ** 0.785 - 0.905
    rho_c = fc_M / (E_M * eps_cr)
    n = 1.0 / (1.0 - rho_c)
    x = np.linspace(1e-9, 5.0, 20000)
    d = np.where(x <= 1.0,
                 1.0 - rho_c * n / (n - 1.0 + x ** n),
                 1.0 - rho_c / (alpha_c * np.maximum(x - 1, 0) ** 2 + x))
    return x * eps_cr, (1.0 - d) * E0 * (x * eps_cr), eps_cr, d


@pytest.mark.parametrize("fc", [20e6, 30e6, 50e6])
def test_gb_compression_peak_and_initial_tangent(fc):
    eps, sig, eps_cr, d = gb_compression_raw(fc, E0)
    assert sig.max() == pytest.approx(fc, rel=1e-4)
    assert eps[np.argmax(sig)] == pytest.approx(eps_cr, rel=2e-3)
    assert d[0] == pytest.approx(0.0, abs=1e-9), "initial tangent must be E0"
    tail = sig[eps > eps_cr]
    assert np.all(np.diff(tail) < 0), "softening branch must decrease"


def test_gb_tension_initial_secant_is_not_E0():
    """Documented property: the ascending branch of the GB tension curve has
    initial secant 1.2*ft/eps_tr, not E0.  That is why gb50010_tension() uses
    only the descending branch."""
    ft_M, E_M = FT / 1e6, E0 / 1e6
    eps_tr = ft_M ** 0.54 * 65e-6
    rho_t = ft_M / (E_M * eps_tr)
    secant0 = 1.2 * rho_t * E0
    # not E0 -- how far off depends on ft (1.02 E0 at ft = 3 MPa, 0.85 at 2 MPa)
    assert abs(secant0 / E0 - 1.0) > 0.01, secant0 / E0


def test_gb_curves_build_and_peak_at_strength():
    c = cv.gb50010_compression(FC, E0)
    t = cv.gb50010_tension(FT, E0)
    # effective stress times (1-d) is the Cauchy stress the curve encodes
    assert (c.sig * (1 - c.dmg)).max() == pytest.approx(FC, rel=5e-3)
    assert (t.sig * (1 - t.dmg)).max() == pytest.approx(FT, rel=1e-6)
    assert t.sigma0 == pytest.approx(FT, rel=1e-9)
    assert c.sigma0 == pytest.approx(0.4 * FC, rel=2e-2)


def test_beta_zero_is_rejected_with_a_reason():
    """beta = 0 leaves CDP with no hardening variable to index its curves by."""
    with pytest.raises(ValueError, match="nothing to index by"):
        cv.gb50010_compression(FC, E0, beta_c=0.0)


def test_small_beta_approaches_pure_damage():
    """As beta falls the plastic strain shrinks and the damage carries more."""
    prev_ep, prev_d = None, None
    for beta in (0.7, 0.3, 0.05):
        c = cv.gb50010_compression(FC, E0, beta_c=beta)
        if prev_ep is not None:
            assert c.ep[-1] < prev_ep
            assert c.dmg.max() > prev_d
        prev_ep, prev_d = c.ep[-1], c.dmg.max()


def test_beta_one_gives_no_damage():
    c = cv.gb50010_compression(FC, E0, beta_c=1.0)
    assert np.abs(c.dmg).max() < 1e-12
    assert c.ep[-1] > 1e-3


@pytest.mark.parametrize("beta", [0.3, 0.7, 0.95])
def test_plastic_fraction_is_what_was_asked_for_where_the_cap_is_slack(beta):
    """ep = beta * inelastic strain, except where the no-expansion cap binds.

    Where it binds the achieved fraction goes UP, not down: the cap lowers
    sigma_bar, which lowers d, which leaves more of the inelastic strain to be
    plastic."""
    c = cv.gb50010_compression(FC, E0, beta_c=beta)
    sig_cauchy = c.sig * (1 - c.dmg)
    eps = c.sig / E0 + c.ep
    e_in = eps - sig_cauchy / E0
    e_in -= e_in[0]
    # strictly before the Cauchy peak the cap cannot have bound yet
    m = (c.ep > 1e-6) & (np.arange(c.ep.size) < int(np.argmax(sig_cauchy)))
    assert m.any()
    assert np.abs(c.ep[m] / (beta * e_in[m]) - 1.0).max() < 1e-6
    # capping sigma_bar lowers d, hence RAISES the plastic fraction actually
    # achieved: the surface shrinks instead of the stiffness degrading
    assert c.beta_effective >= beta - 1e-9


# --------------------------------------------------------------------------- #


def test_abaqus_table_round_trip():
    """sigma reconstructed from the converted curve must match the input table."""
    e_in = np.array([0.0, 2e-4, 6e-4, 1.2e-3, 2.5e-3])
    sig = np.array([20e6, 27e6, 30e6, 24e6, 12e6])
    d = np.array([0.0, 0.0, 0.05, 0.30, 0.70])
    c = cv.from_abaqus_tables(sig, e_in, E0, damage=d)
    assert np.allclose(c.sig, sig / (1 - d))
    assert np.allclose(c.dmg, d)
    assert np.allclose(c.ep, e_in - d / (1 - d) * sig / E0)
    # and the total strain implied is the elastic part plus the inelastic table
    eps = c.sig / E0 + c.ep
    assert np.allclose(eps - sig / E0, e_in)


def test_negative_plastic_strain_is_rejected_with_a_useful_message():
    """The classic Abaqus CDP input error, caught at setup."""
    e_in = np.array([0.0, 1e-4, 2e-4])
    sig = np.array([30e6, 28e6, 26e6])
    d = np.array([0.0, 0.6, 0.95])          # damage far too aggressive
    with pytest.raises(ValueError, match="not increasing"):
        cv.from_abaqus_tables(sig, e_in, E0, damage=d)


def test_fracture_energy_curve_dissipates_gf():
    Gf, l_ref = 120.0, 0.2
    c = cv.from_fracture_energy(FT, Gf, E0, l_ref, beta_t=1.0)
    sig = c.sig * (1 - c.dmg)
    w = c.ep * l_ref                        # beta = 1 -> ep is the whole inelastic strain
    assert np.trapezoid(sig, w) == pytest.approx(Gf, rel=1e-3)


@pytest.mark.parametrize("shape", ["exponential", "linear"])
def test_softening_shapes_are_monotone(shape):
    c = cv.from_fracture_energy(FT, 120.0, E0, 0.2, shape=shape)
    assert np.all(np.diff(c.sig) <= 1e-9), "effective stress must not expand"
    assert c.sigma0 == pytest.approx(FT, rel=1e-9)


@pytest.mark.parametrize("beta", [0.3, 0.7, 0.9, 1.0])
def test_effective_stress_never_expands_after_the_peak(beta):
    """The yield surface must not grow while the material softens."""
    for c in (cv.gb50010_tension(FT, E0, beta_t=beta),
              cv.gb50010_compression(FC, E0, beta_c=beta),
              cv.from_fracture_energy(FT, 120.0, E0, 0.2, beta_t=beta)):
        i0 = int(np.argmax(c.sig))
        assert np.all(np.diff(c.sig[i0:]) <= 1e-6), c.name


# --------------------------------------------------------------------------- #


def test_crack_band_scaling_is_a_pure_change_of_variable():
    """evaluate(ep, k) must equal evaluate(ep*k, 1) with the slope times k."""
    c = cv.gb50010_tension(FT, E0)
    ep = np.linspace(0.0, c.ep[-1] * 0.8, 500)
    for k in (0.5, 1.0, 2.5):
        s1, d1, m1 = c.evaluate(ep, k)
        s2, d2, m2 = c.evaluate(ep * k, 1.0)
        assert np.allclose(s1, s2)
        assert np.allclose(d1, d2 * k)
        assert np.allclose(m1, m2)


def test_curve_holds_flat_past_the_table():
    """Extrapolating a softening branch to negative stress must not happen."""
    c = cv.gb50010_tension(FT, E0)
    s, slope, d = c.evaluate(np.array([c.ep[-1] * 10.0]))
    assert s[0] == pytest.approx(c.sig[-1])
    assert slope[0] == 0.0
    assert s[0] > 0.0


def test_curve_rejects_bad_input():
    with pytest.raises(ValueError, match="start at ep = 0"):
        cv.Curve(np.array([1e-4, 2e-4]), np.array([1.0, 2.0]), np.zeros(2))
    with pytest.raises(ValueError, match="damage must lie"):
        cv.Curve(np.array([0.0, 1e-4]), np.array([1.0, 2.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="stress must be positive"):
        cv.Curve(np.array([0.0, 1e-4]), np.array([1.0, -2.0]), np.zeros(2))
