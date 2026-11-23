"""Constitutive verification -- the tests that pin down P1, P2 and P3.

Pure shear is used as the primary check because it is an EXACT strain-driven
state: with only gamma_xy imposed, the stress stays pure shear, the flow
direction has only a shear component, and the response has a closed form.

Writing the Chaboche model in equivalent (uniaxial) variables, define
x_i = sqrt(3) * X_i,12.  Then dx_i/dp = C_i - gamma_i x_i, exactly as in
uniaxial tension, so under monotonic loading from a virgin state

    sqrt(3) tau(p) = sy0 + Q(1 - e^-bp) + H p + sum_i (C_i/g_i)(1 - e^-g_i p)
    gamma(p)       = tau/G + sqrt(3) p

Both the factor-of-two Voigt error (P1) and the 2G-instead-of-3G plastic
multiplier (P2) break this identity, and by different amounts.
"""

import numpy as np
import pytest

from vfife.materials import create

PROPS = dict(
    E=210e9, nu=0.3, rho=7800.0,
    yield_strength=250e6,
    Q_inf=100e6, b=20.0, H_iso=1.0e9,
    C1=30e9, gamma1=300.0, C2=5e9, gamma2=30.0,
)
G = PROPS["E"] / (2 * (1 + PROPS["nu"]))
SQ3 = np.sqrt(3.0)


def sigma_y(p):
    return (
        PROPS["yield_strength"]
        + PROPS["Q_inf"] * (1 - np.exp(-PROPS["b"] * p))
        + PROPS["H_iso"] * p
    )


def tau_exact(p):
    x = (PROPS["C1"] / PROPS["gamma1"]) * (1 - np.exp(-PROPS["gamma1"] * p)) + (
        PROPS["C2"] / PROPS["gamma2"]
    ) * (1 - np.exp(-PROPS["gamma2"] * p))
    return (sigma_y(p) + x) / SQ3


def drive_shear(mat, gamma_path, dt=1.0):
    """Feed a gamma_xy path through the material. Returns tau and state history."""
    state = mat.init_state(1)
    eps = np.zeros((1, 6))
    taus, states = [], []
    for g in gamma_path:
        eps[0, 3] = g
        sig = mat.update(eps, state, dt)
        taus.append(float(sig[0, 3]))
        states.append(state.copy())
    return np.array(taus), np.concatenate(states, axis=0)


@pytest.mark.parametrize("n_steps", [2000, 4000])
def test_pure_shear_monotonic_matches_closed_form(n_steps):
    mat = create("chaboche", n_newton=4, **PROPS)
    p = np.linspace(0.0, 0.02, n_steps)[1:]
    gamma = tau_exact(p) / G + SQ3 * p
    taus, _ = drive_shear(mat, gamma)

    ref = tau_exact(p)
    plastic = p > 1e-4                     # skip the first few tiny increments
    rel = np.abs(taus[plastic] - ref[plastic]) / ref[plastic]
    assert rel.max() < 2e-3, f"max relative error {rel.max():.2e}"


def test_pure_shear_error_halves_with_step_size():
    """Backward Euler is first order in dp; confirm the observed order."""
    errs = []
    for n in (1000, 2000, 4000):
        mat = create("chaboche", n_newton=4, **PROPS)
        p = np.linspace(0.0, 0.02, n)[1:]
        gamma = tau_exact(p) / G + SQ3 * p
        taus, _ = drive_shear(mat, gamma)
        m = p > 1e-3
        errs.append(np.abs(taus[m] - tau_exact(p)[m]).max())
    assert errs[1] < 0.62 * errs[0] and errs[2] < 0.62 * errs[1], errs


def test_voigt_convention_engineering_shear():
    """P1: eps_p must be stored in ENGINEERING shear.

    gamma_total = tau/G + gamma_plastic must hold exactly.  Storing the tensor
    shear (the original bug) makes the recovered plastic shear half as large.
    """
    mat = create("chaboche", n_newton=4, **PROPS)
    p = np.linspace(0.0, 0.01, 2000)[1:]
    gamma = tau_exact(p) / G + SQ3 * p
    taus, states = drive_shear(mat, gamma)

    gamma_p = states[:, 3]                     # eps_p component 4 (engineering)
    recovered = gamma - taus / G
    m = p > 1e-3
    assert np.abs(gamma_p[m] / recovered[m] - 1.0).max() < 1e-6


def test_equivalent_plastic_strain_consistency():
    """alpha must equal p = gamma_p / sqrt(3) for pure shear."""
    mat = create("chaboche", n_newton=4, **PROPS)
    p = np.linspace(0.0, 0.01, 2000)[1:]
    gamma = tau_exact(p) / G + SQ3 * p
    _, states = drive_shear(mat, gamma)
    alpha, gamma_p = states[:, 6], states[:, 3]
    m = p > 1e-3
    assert np.abs(alpha[m] * SQ3 / gamma_p[m] - 1.0).max() < 1e-6


def test_plastic_multiplier_denominator_is_3G():
    """P2: the first plastic increment pins down the 3G in the denominator.

    Starting from the virgin state (X = 0), the consistency condition gives
        dp = f_trial / (3G + C1 + C2 + H')
    Using 2G, as the original did, is a 33% error in the very first step.
    """
    mat = create("chaboche", n_newton=6, **PROPS)
    state = mat.init_state(1)
    gy = sigma_y(0.0) / (SQ3 * G)              # gamma at first yield
    eps = np.zeros((1, 6))
    eps[0, 3] = gy
    mat.update(eps, state, 1.0)                # sit exactly at yield
    state0 = state.copy()

    dg = 1e-5
    eps[0, 3] = gy + dg
    mat.update(eps, state, 1.0)
    dp = float(state[0, 6] - state0[0, 6])

    f_trial = SQ3 * G * dg
    expected = f_trial / (
        3 * G + PROPS["C1"] + PROPS["C2"]
        + PROPS["Q_inf"] * PROPS["b"] + PROPS["H_iso"]
    )
    assert abs(dp / expected - 1.0) < 5e-3, (dp, expected)


def test_bauschinger_reversal():
    """Reversing the shear must reproduce the 1-D equivalent Chaboche model."""
    mat = create("chaboche", n_newton=4, **PROPS)
    n = 3000
    g_peak = 0.02
    path = np.concatenate([
        np.linspace(0, g_peak, n)[1:],
        np.linspace(g_peak, -g_peak, 2 * n)[1:],
    ])
    taus, states = drive_shear(mat, path)
    ref = _shear_reference(path)
    scale = np.abs(ref).max()
    assert np.abs(taus - ref).max() / scale < 5e-3


def _shear_reference(gamma_path):
    """Independent 1-D equivalent-variable integration of the same model."""
    C1, g1 = PROPS["C1"], PROPS["gamma1"]
    C2, g2 = PROPS["C2"], PROPS["gamma2"]
    tau = 0.0
    x1 = x2 = 0.0
    p = 0.0
    g_prev = 0.0
    out = []
    for g in gamma_path:
        tau_tr = tau + G * (g - g_prev)
        g_prev = g
        xi = SQ3 * tau_tr - (x1 + x2)
        s = 1.0 if xi >= 0 else -1.0
        if abs(xi) - sigma_y(p) > 0:
            dp = 0.0
            for _ in range(30):
                d1, d2 = 1 + g1 * dp, 1 + g2 * dp
                R = (
                    s * SQ3 * tau_tr - 3 * G * dp
                    - s * ((x1 + s * C1 * dp) / d1 + (x2 + s * C2 * dp) / d2)
                    - sigma_y(p + dp)
                )
                dR = (
                    -3 * G
                    - s * ((s * C1 - g1 * x1) / d1 ** 2 + (s * C2 - g2 * x2) / d2 ** 2)
                    - (PROPS["Q_inf"] * PROPS["b"] * np.exp(-PROPS["b"] * (p + dp))
                       + PROPS["H_iso"])
                )
                step = R / dR
                dp = max(dp - step, 0.0)
                if abs(step) < 1e-16:
                    break
            x1 = (x1 + s * C1 * dp) / (1 + g1 * dp)
            x2 = (x2 + s * C2 * dp) / (1 + g2 * dp)
            p += dp
            tau = tau_tr - SQ3 * G * s * dp
        else:
            tau = tau_tr
        out.append(tau)
    return np.array(out)


def test_damage_applied_once():
    """P3: Cauchy stress = (1-D) * effective stress, not (1-D)^2."""
    props = dict(PROPS, damage_initiation_strain=0.002,
                 damage_evolution_rate=20.0, damage_max=0.9)
    mat_d = create("chaboche", n_newton=4, **props)
    mat_0 = create("chaboche", n_newton=4, **PROPS)

    p = np.linspace(0.0, 0.02, 3000)[1:]
    gamma = tau_exact(p) / G + SQ3 * p
    tau_d, st_d = drive_shear(mat_d, gamma)
    tau_0, _ = drive_shear(mat_0, gamma)

    D = st_d[:, 19]
    m = (D > 1e-6) & (D < 0.85)
    assert m.any()
    # damaged stress must track (1-D) * undamaged, to within the small change
    # the damage itself makes to the plastic path
    ratio = tau_d[m] / (tau_0[m] * (1 - D[m]))
    assert np.abs(ratio - 1.0).max() < 0.05, np.abs(ratio - 1.0).max()
    # and it must NOT look like (1-D)^2
    ratio2 = tau_d[m] / (tau_0[m] * (1 - D[m]) ** 2)
    assert np.abs(ratio2 - 1.0).max() > 0.10


def test_viscous_damage_regularisation():
    """Duvaut-Lions viscosity must delay D without changing the rate-independent D_bar."""
    base = dict(PROPS, damage_initiation_strain=0.002,
                damage_evolution_rate=20.0, damage_max=0.9)
    p = np.linspace(0.0, 0.02, 2000)[1:]
    gamma = tau_exact(p) / G + SQ3 * p

    dt = 1e-3
    T = len(gamma) * dt                       # total loading time
    res = {}
    for eta in (0.0, 0.2 * T, 2.0 * T):
        mat = create("chaboche", n_newton=4, damage_viscosity=eta, **base)
        _, st = drive_shear(mat, gamma, dt=dt)
        res[eta] = (st[-1, 19], st[-1, 20])   # (D, D_bar)

    D0, Dbar0 = res[0.0]
    # the rate-independent solution must be untouched by the regularisation
    for eta, (D, Dbar) in res.items():
        assert abs(Dbar - Dbar0) < 1e-12, "viscosity changed the inviscid solution"
    assert D0 == pytest.approx(Dbar0)
    # and D must lag more as eta grows
    assert D0 > res[0.2 * T][0] > res[2.0 * T][0] > 0.0
    assert res[2.0 * T][0] / D0 < 0.6


def test_viscosity_vanishes_for_slow_loading():
    """As the loading time grows relative to eta, D must approach D_bar."""
    base = dict(PROPS, damage_initiation_strain=0.002,
                damage_evolution_rate=20.0, damage_max=0.9)
    p = np.linspace(0.0, 0.02, 2000)[1:]
    gamma = tau_exact(p) / G + SQ3 * p
    eta = 1.0
    ratios = []
    for dt in (1e-3, 1e-2, 1e-1):             # same path, ever slower
        mat = create("chaboche", n_newton=4, damage_viscosity=eta, **base)
        _, st = drive_shear(mat, gamma, dt=dt)
        ratios.append(st[-1, 19] / st[-1, 20])
    assert ratios[0] < ratios[1] < ratios[2]
    assert ratios[2] > 0.97


def test_uniaxial_stress_matches_closed_form():
    """Uniaxial tension with lateral stresses relaxed to zero."""
    mat = create("chaboche", n_newton=4, **PROPS)
    E, nu = PROPS["E"], PROPS["nu"]
    state = mat.init_state(1)
    eps = np.zeros((1, 6))
    lat = 0.0
    p_target = np.linspace(0.0, 0.02, 2000)[1:]
    axial = tau_exact(p_target) * SQ3 / E + p_target      # sigma/E + p
    sigs, ps = [], []
    for ax in axial:
        eps[0, 0] = ax
        # Newton on the lateral strain so that syy = szz = 0
        for _ in range(6):
            eps[0, 1] = eps[0, 2] = lat
            trial = state.copy()
            sig = mat.update(eps, trial, 1.0)
            r = sig[0, 1]
            if abs(r) < 1e-3:
                break
            # d syy / d lat ~ (E/(1+nu))/(1-...) ; a secant-free estimate is fine
            lat -= r / (E / (1 + nu) / (1 - nu))
        eps[0, 1] = eps[0, 2] = lat
        sig = mat.update(eps, state, 1.0)
        sigs.append(float(sig[0, 0]))
        ps.append(float(state[0, 6]))

    sigs, ps = np.array(sigs), np.array(ps)
    ref = tau_exact(ps) * SQ3
    m = ps > 1e-3
    assert np.abs(sigs[m] / ref[m] - 1.0).max() < 5e-3
