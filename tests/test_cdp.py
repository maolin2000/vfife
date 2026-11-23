"""Concrete damaged plasticity verification.

The tests fall into three groups:

  SURFACE   fb0/fc0 and Kc are statements about the SHAPE of the yield surface,
            so they are checked on the surface directly (via `yield_function`)
            rather than inferred from a loading path.
  PATH      uniaxial tension and compression must reproduce the curves the
            model was built from, exactly -- that is the definition of a
            correctly calibrated CDP.  Reversal tests pin the stiffness
            recovery factors.
  OBJECTIVITY  the crack band must make the softening response independent of
            element size, expressed as: the sigma-vs-crack-opening curve is
            the same for every element length.
"""

import functools

import numpy as np
import pytest

from vfife.materials import create

E, NU, RHO = 3.0e10, 0.2, 2400.0
FC, FT = 30.0e6, 3.0e6
GF = 120.0
L_REF = 0.2


def cdp(**kw):
    kw.setdefault("compression", {"source": "gb50010", "fc": FC})
    kw.setdefault("tension", {"source": "gb50010", "ft": FT})
    kw.setdefault("l_ref", L_REF)
    return create("cdp", E=E, nu=NU, rho=RHO, **kw)


# Lateral tolerance: 1 kPa is 3e-4 of ft.  Tightening it further only buys
# Newton iterations -- and every iteration is a full material update, which at
# one integration point is dominated by NumPy call overhead (0.6 ms/call
# measured, against 1.6 us/point when batched).  Batching the drivers and
# loosening this tolerance is what keeps this file inside a CI budget.
LAT_TOL = 1.0e3
LAT_TANGENT = E / (1 + NU) / (1 - NU)


def uniaxial_batch(mat, eps_paths, n_lat=8, dt=1.0, st=None, lat=None,
                   lat_tol=LAT_TOL):
    """Drive N independent uniaxial-STRESS paths at once.

    eps_paths : (n_steps, n_points) axial strain history per point.
    Pass `st`/`lat` back in to continue a path (needed for the reversal tests,
    where the probe leg must also be uniaxial STRESS -- otherwise the lateral
    plastic strain left by the first leg leaks into the axial stress).

    The lateral strain is found by a SECANT iteration, not by Newton on the
    elastic tangent: once the material softens, the true lateral tangent is far
    from the elastic one and a fixed-tangent Newton needs ~30 evaluations per
    step and still leaves a few hundred Pa.  The secant gets under 1 Pa in two
    or three, which matters because several tests check quantities (r, and
    therefore the stiffness-recovery factors) that are only exact when the
    lateral stress is exactly zero.

    Returns (sigma_axial (n_steps, n_points), state, lateral strain).
    """
    eps_paths = np.atleast_2d(eps_paths)
    npt = eps_paths.shape[1]
    if st is None:
        st = mat.init_state(npt)
    e = np.zeros((npt, 6))
    lat = np.zeros(npt) if lat is None else np.array(lat, dtype=float)
    out = np.empty_like(eps_paths)

    def lateral_stress(x):
        e[:, 1] = e[:, 2] = x
        return mat.update(e, st.copy(), dt)[:, 1].copy()

    for i, ax in enumerate(eps_paths):
        e[:, 0] = ax
        x0 = lat.copy()
        g0 = lateral_stress(x0)
        x1 = x0 - g0 / LAT_TANGENT
        for _ in range(n_lat):
            g1 = lateral_stress(x1)
            if np.abs(g1).max() < lat_tol:
                break
            dg = g1 - g0
            ok = np.abs(dg) > 1e-30
            step = np.where(ok, g1 * (x1 - x0) / np.where(ok, dg, 1.0),
                            g1 / LAT_TANGENT)
            x0, g0 = x1, g1
            x1 = x1 - step
        lat = x1
        e[:, 1] = e[:, 2] = lat
        s = mat.update(e, st, dt)
        out[i] = s[:, 0]
    return out, st, lat


def uniaxial(mat, eps_path, **kw):
    """Single-point convenience wrapper. Returns (sigma, ept, epc, d) per step."""
    sig, st, _ = uniaxial_batch(mat, np.asarray(eps_path)[:, None], **kw)
    n = sig.shape[0]
    return np.column_stack([
        sig[:, 0],
        np.full(n, st[0, 6]), np.full(n, st[0, 7]), np.full(n, st[0, 14]),
    ])


# --------------------------------------------------------------------------- #
#  SURFACE
# --------------------------------------------------------------------------- #


def _root_scale(mat, direction, ept=0.0, epc=0.0, lo=1.0, hi=1e10):
    """Find s such that F(s * direction) = 0. Bisection; the surface is convex."""
    d = np.asarray(direction, float)
    f = lambda s: float(mat.yield_function((s * d)[None, :], ept, epc)[0])
    assert f(lo) < 0 < f(hi), (f(lo), f(hi))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@pytest.mark.parametrize("ratio", [1.10, 1.16, 1.25])
def test_equibiaxial_compressive_strength(ratio):
    """The surface must yield in equibiaxial compression at fb0 = ratio * fc0."""
    m = cdp(fb0_fc0=ratio)
    uni = _root_scale(m, [-1, 0, 0, 0, 0, 0])
    bi = _root_scale(m, [-1, -1, 0, 0, 0, 0])
    # bisection on F, whose magnitude is ~1e7 Pa: 1e-6 is the double-precision floor
    assert bi / uni == pytest.approx(ratio, rel=1e-6)


def test_uniaxial_strengths_are_the_curve_values():
    """Uniaxial tension and compression must yield exactly at st0 and sc0."""
    m = cdp()
    assert _root_scale(m, [-1, 0, 0, 0, 0, 0]) == pytest.approx(
        m.curve_c.sigma0, rel=1e-6)
    assert _root_scale(m, [1, 0, 0, 0, 0, 0]) == pytest.approx(
        m.curve_t.sigma0, rel=1e-6)


@pytest.mark.parametrize("Kc", [0.55, 2.0 / 3.0, 0.8])
def test_meridian_ratio_is_Kc(Kc):
    """q on the tensile meridian over q on the compressive meridian = Kc.

    Measured deep in triaxial compression, where sig_max < 0 and the gamma term
    (the only place Kc enters) is active.
    """
    m = cdp(Kc=Kc)
    p = 5.0 * FC                                    # well inside compression
    # tensile meridian: s1 > s2 = s3  ->  s = (2q/3, -q/3, -q/3) about -p
    # compressive meridian: s1 = s2 > s3  ->  s = (q/3, q/3, -2q/3)
    def q_on(sdev):
        sdev = np.asarray(sdev, float)
        f = lambda q: float(m.yield_function(
            (np.array([-p, -p, -p, 0, 0, 0]) + q * sdev)[None, :])[0])
        lo, hi = 1e-3, 50.0 * FC
        assert f(lo) < 0 < f(hi)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
        return 0.5 * (lo + hi)

    q_t = q_on([2 / 3, -1 / 3, -1 / 3, 0, 0, 0])
    q_c = q_on([1 / 3, 1 / 3, -2 / 3, 0, 0, 0])
    assert q_t / q_c == pytest.approx(Kc, rel=1e-6)


# --------------------------------------------------------------------------- #
#  PATH
# --------------------------------------------------------------------------- #


FINE_T = {"source": "gb50010", "ft": FT, "n_points": 1600}
FINE_C = {"source": "gb50010", "fc": FC, "n_points": 1600}


def test_uniaxial_tension_reproduces_input_curve():
    m = cdp(tension=FINE_T)
    c = m.curve_t
    eps_ref = c.sig / E + c.ep
    sig_ref = c.sig * (1.0 - c.dmg)
    path = np.linspace(0.0, eps_ref[-1] * 0.9, 900)[1:]
    res = uniaxial(cdp(tension=FINE_T), path)
    exp = np.interp(path, eps_ref, sig_ref)
    soft = path > FT / E * 1.05
    # the peak is sampled, not solved for: softening begins immediately at ft,
    # so the largest sampled stress sits just below it
    assert res[:, 0].max() == pytest.approx(FT, rel=1e-2)
    assert np.abs(res[soft, 0] - exp[soft]).max() / FT < 5e-3


def test_uniaxial_compression_reproduces_input_curve():
    m = cdp(compression=FINE_C)
    c = m.curve_c
    eps_ref = -(c.sig / E + c.ep)
    sig_ref = -c.sig * (1.0 - c.dmg)
    path = np.linspace(0.0, eps_ref[-1] * 0.9, 3000)[1:]
    res = uniaxial(cdp(compression=FINE_C), path)
    exp = np.interp(-path, -eps_ref, sig_ref)
    plastic = -path > m.curve_c.sigma0 / E * 1.2
    assert -res[:, 0].min() == pytest.approx(FC, rel=1e-2)
    assert np.abs(res[plastic, 0] - exp[plastic]).max() / FC < 5e-3


def test_return_map_error_shrinks_with_step_size():
    """Confirm the residual is discretisation, not a modelling error."""
    c = cdp(tension=FINE_T).curve_t
    eps_ref = c.sig / E + c.ep
    sig_ref = c.sig * (1.0 - c.dmg)
    errs = []
    for n in (250, 1000, 4000):
        path = np.linspace(0.0, eps_ref[-1] * 0.6, n)[1:]
        # a tight lateral tolerance here: the default 1 kPa puts a floor of
        # ~5 kPa on the axial stress, which masks the convergence being measured
        res = uniaxial(cdp(tension=FINE_T), path, lat_tol=1.0)
        exp = np.interp(path, eps_ref, sig_ref)
        soft = path > FT / E * 1.1
        errs.append(np.abs(res[soft, 0] - exp[soft]).max())
    assert errs[1] < 0.5 * errs[0] and errs[2] < 0.5 * errs[1], errs


def test_hardening_variables_separate_tension_and_compression():
    """Tension must not accumulate compressive hardening, and vice versa."""
    c = cdp().curve_t
    # r is exactly 1 (tension) or 0 (compression) only if the lateral stress is
    # exactly zero, so the cross term is driver noise, not model behaviour --
    # hence a tight lateral tolerance and a "negligible", not "zero", assertion
    kw = dict(lat_tol=1.0)
    t = uniaxial(cdp(), np.linspace(0, (c.sig / E + c.ep)[-1] * 0.5, 400)[1:], **kw)
    assert t[-1, 1] > 1e-5
    assert t[-1, 2] < 1e-3 * t[-1, 1], (t[-1, 1], t[-1, 2])

    cc = cdp().curve_c
    k = uniaxial(cdp(), np.linspace(0, -(cc.sig / E + cc.ep)[-1] * 0.5, 400)[1:], **kw)
    assert k[-1, 2] > 1e-5
    assert k[-1, 1] < 1e-3 * k[-1, 2], (k[-1, 2], k[-1, 1])


def _reversal(wc, wt):
    """Damage in tension, unload to zero STRESS, then probe in compression.

    Unloading to zero total strain would be wrong, and instructively so: after
    6.5e-4 of tensile plastic strain, forcing the total strain back to zero
    means closing the crack completely, which takes 19 MPa of compression --
    past sigma_c0 = 12 MPa.  The specimen then crushes, and the "stiffness
    recovery" being measured is contaminated by fresh compressive damage.  Zero
    stress is at eps = eps_p, and the probe runs from there.
    """
    m = cdp(wc=wc, wt=wt)
    peak = (m.curve_t.sig / E + m.curve_t.ep)[-1] * 0.25
    kw = dict(lat_tol=1.0)

    _, st, lat = uniaxial_batch(m, np.linspace(0.0, peak, 600)[1:, None], **kw)
    d_after_tension = float(st[0, 14])

    ep0 = float(st[0, 8])
    _, st, lat = uniaxial_batch(m, np.linspace(peak, ep0, 200)[1:, None],
                                st=st, lat=lat, **kw)

    ep1 = float(st[0, 8])
    probe = ep1 - 0.2 * m.curve_c.sigma0 / E
    sig, st, lat = uniaxial_batch(m, np.linspace(ep1, probe, 200)[1:, None],
                                  st=st, lat=lat, **kw)
    assert float(st[0, 7]) < 1e-9, "the probe must stay elastic in compression"
    E_sec = float(sig[-1, 0]) / (probe - float(st[0, 8]))
    return d_after_tension, E_sec, float(st[0, 14])


def test_compressive_stiffness_recovery():
    """wc = 1 (Abaqus default) recovers the compressive stiffness after cracking."""
    d_t, E_rec, d_rev = _reversal(wc=1.0, wt=0.0)
    assert d_t > 0.2, "the tension leg should have caused real damage"
    # r is 0 only when the stress state is exactly uniaxial compression, so a
    # residue of order the lateral tolerance survives; it must be negligible
    assert d_rev < 1e-3 * d_t, (d_t, d_rev)
    assert E_rec == pytest.approx(E, rel=2e-3)


def test_no_recovery_when_wc_zero():
    """wc = 0 keeps the tensile damage active in compression."""
    d_t, E_rec, d_rev = _reversal(wc=0.0, wt=0.0)
    assert d_rev == pytest.approx(d_t, rel=1e-2)
    # the secant modulus tracks the CURRENT damage; d drifts ~1% during the
    # unload leg because the last tension step leaves F marginally above zero
    assert E_rec == pytest.approx(E * (1.0 - d_rev), rel=2e-3)


@pytest.mark.parametrize("psi", [10.0, 36.0, 50.0])
def test_dilation_angle_controls_plastic_volume_change(psi):
    """Plastic volumetric strain per unit plastic multiplier is tan(psi)."""
    m = cdp(dilation_angle=psi)
    c = m.curve_c
    path = np.linspace(0.0, -(c.sig / E + c.ep)[-1] * 0.5, 500)[1:]
    _, st, _ = uniaxial_batch(m, path[:, None])
    ep = st[0, 0:6]
    vol = ep[0] + ep[1] + ep[2]
    assert vol > 0.0, "concrete must dilate plastically in compression"
    # tr(m) = tan(psi) and the axial component dominates, so the ratio
    # vol / |axial plastic strain| grows monotonically with tan(psi)
    ratio = vol / abs(ep[0])
    assert ratio == pytest.approx(_expected_dilation_ratio(np.tan(np.deg2rad(psi))),
                                  rel=0.05), ratio


def _expected_dilation_ratio(tanpsi):
    """tr(m)/m_1 for uniaxial compression, where s/q = (-2/3, 1/3, 1/3)."""
    m1 = -1.0 + tanpsi / 3.0                # dG/dq ~ 1 far from the apex
    return tanpsi / abs(m1)


# --------------------------------------------------------------------------- #
#  OBJECTIVITY
# --------------------------------------------------------------------------- #


# The snap-back limit for ft = 3 MPa, Gf = 120 J/m2, E = 30 GPa is 0.283 m --
# lower than the textbook E*Gf/ft^2 = 0.4 m, because the EFFECTIVE-stress
# curve is steeper than sigma(w) once beta_t and the damage are folded in.
# bind_points() refuses anything coarser, so stay under it.
SIZES = np.array([0.05, 0.1, 0.2, 0.25])


@functools.lru_cache(maxsize=2)
def _crack_band_curves(n=4000):
    """sigma vs crack opening for several element sizes, in ONE batched run.

    Only bind_points() differs between the points, so they share a material
    object and go through the solver as a single 4-point batch -- 4x fewer
    NumPy calls, which is the whole cost at this size.
    """
    m = cdp(tension={"source": "fracture_energy", "ft": FT, "Gf": GF},
            crack_band="tension")
    m.bind_points(SIZES)
    w_end = 8.0 * GF / FT
    paths = np.stack([np.linspace(0.0, FT / E + w_end / l, n)[1:] for l in SIZES],
                     axis=1)
    sig, _, _ = uniaxial_batch(m, paths)
    w = (paths - sig / E) * SIZES[None, :]
    return w, sig


@pytest.mark.parametrize("i", range(len(SIZES)))
def test_crack_band_makes_softening_mesh_independent(i):
    """sigma vs crack OPENING must be the same curve for every element size."""
    w, sig = _crack_band_curves()
    ref = list(SIZES).index(L_REF)
    # start past w = 0: right at the peak the curve is nearly vertical, so
    # interpolating there compares sampling noise, not the softening law
    grid = np.linspace(0.05 * GF / FT, 4.0 * GF / FT, 200)
    a = np.interp(grid, w[:, ref], sig[:, ref])
    b = np.interp(grid, w[:, i], sig[:, i])
    assert np.abs(a - b).max() / FT < 0.03, np.abs(a - b).max() / FT


def test_fracture_energy_is_dissipated():
    """Area under sigma vs crack opening equals Gf."""
    w, sig = _crack_band_curves()
    ref = list(SIZES).index(L_REF)
    assert np.trapezoid(sig[:, ref], w[:, ref]) == pytest.approx(GF, rel=0.06)


def test_without_crack_band_the_response_is_mesh_dependent():
    """The control: turn the crack band off and objectivity must be lost."""
    m = cdp(tension={"source": "fracture_energy", "ft": FT, "Gf": GF},
            crack_band="none")
    m.bind_points(np.array([0.05, 0.5]))
    path = np.linspace(0.0, FT / E + 8 * GF / FT / L_REF, 2000)[1:]
    sig, _, _ = uniaxial_batch(m, np.repeat(path[:, None], 2, axis=1))
    energy = np.trapezoid(sig, path, axis=0) * np.array([0.05, 0.5])
    assert abs(energy[0] / energy[1] - 1.0) > 0.5


# --------------------------------------------------------------------------- #
#  Viscous regularisation
# --------------------------------------------------------------------------- #


def test_viscosity_delays_damage_and_vanishes_for_slow_loading():
    c = cdp().curve_t
    path = np.linspace(0.0, (c.sig / E + c.ep)[-1] * 0.5, 500)[1:]
    out = {}
    for mu, dt in ((0.0, 1e-3), (1e-2, 1e-3), (1e-1, 1e-3), (1e-2, 1e-1)):
        m = cdp(viscosity=mu)
        st = m.init_state(1)
        e = np.zeros((1, 6))
        for ax in path:
            e[0, 0] = ax
            m.update(e, st, dt)
        out[(mu, dt)] = float(st[0, 14])
    assert out[(0.0, 1e-3)] > out[(1e-2, 1e-3)] > out[(1e-1, 1e-3)] > 0.0
    # same mu, 100x slower loading -> back to the rate-independent answer
    assert out[(1e-2, 1e-1)] == pytest.approx(out[(0.0, 1e-3)], rel=0.02)


def test_zero_viscosity_is_exactly_rate_independent():
    c = cdp().curve_t
    path = np.linspace(0.0, (c.sig / E + c.ep)[-1] * 0.4, 250)[1:]
    res = []
    for dt in (1e-6, 1.0, 1e6):
        m = cdp(viscosity=0.0)
        st = m.init_state(1)
        e = np.zeros((1, 6))
        for ax in path:
            e[0, 0] = ax
            s = m.update(e, st, dt)
        res.append(float(s[0, 0]))
    assert res[0] == res[1] == res[2]
