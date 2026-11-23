"""The top frequency for the stiffness-proportional damping bound, by power
iteration instead of Gershgorin, and Rayleigh stated by ratios at two
frequencies."""
import math

import numpy as np
import pytest
import scipy.sparse as sp

from vfife.elements.stability import damped_step_factor, power_omega_max
from vfife.io.schema import DampingSpec


def test_power_iteration_finds_the_top_frequency_of_a_random_pair():
    rng = np.random.default_rng(1)
    n = 60
    B = rng.standard_normal((n, n))
    K = sp.csr_matrix(B @ B.T + n * np.eye(n))
    m = rng.uniform(0.5, 3.0, n)
    free = np.ones(n, bool); free[:5] = False
    idx = np.flatnonzero(free)
    d = 1 / np.sqrt(m[idx])
    exact = np.sqrt(np.linalg.eigvalsh((d[:, None] * K.toarray()[np.ix_(idx, idx)]) * d[None, :]).max())
    got = power_omega_max(K, m, free, iterations=2000, tol=1e-9)
    assert got == pytest.approx(exact, rel=1e-4)
    assert got <= exact * (1 + 1e-9)                      # from below, by construction
    assert power_omega_max(K, m, np.zeros(n, bool)) == 0.0


def test_the_beta_bound_uses_the_true_top_frequency_and_the_undamped_step_does_not_move():
    import sys; sys.path.insert(0, "tests")
    from test_rest_stays_at_rest import FAMILIES, _build, _strong_beta, _kick, _run_n, _displacement
    fid, section, mesh, material = next(f for f in FAMILIES if f[0] == "shell4")
    undamped = _build(section, mesh, material, {"type": "none"}, total_time=1e-6)
    assert not hasattr(undamped, "omega_max_sharp")
    alpha_only = _build(section, mesh, material, {"type": "rayleigh", "alpha": 5.0}, total_time=1e-6)
    assert not hasattr(alpha_only, "omega_max_sharp")
    assert alpha_only.stable_dt / alpha_only.damping_step_factor == undamped.stable_dt
    beta = _strong_beta(section, mesh, material)
    s = _build(section, mesh, material, {"type": "rayleigh", "alpha": 0.0, "beta": beta})
    # the sharp frequency is the pair's top eigenvalue, under the Gershgorin bound
    idx = np.flatnonzero(~s._held_dofs())
    d = 1 / np.sqrt(s.mass[idx])
    exact = np.sqrt(np.linalg.eigvalsh((d[:, None] * sp.csr_matrix(s.K)[idx][:, idx].toarray()) * d[None, :]).max())
    assert s.omega_max_sharp == pytest.approx(exact, rel=1e-3)
    assert s.omega_max_sharp < s.omega_max_gershgorin / 2          # the shell's bound is loose
    # the step is the one the sharp frequency allows, larger than Gershgorin's
    assert s.damping_step_factor == pytest.approx(damped_step_factor(1.05 * s.omega_max_sharp, beta=beta))
    assert s.damping_step_factor > 2 * damped_step_factor(s.omega_max_gershgorin, beta=beta)
    # and at that larger step a kicked model still stays at rest
    _kick(s); _run_n(s, 600)
    assert float(np.abs(_displacement(s)).max()) < 1e-9


def test_rayleigh_by_ratio_at_two_frequencies():
    w1, w2 = 2 * math.pi * 0.5, 2 * math.pi * 8.0
    d = DampingSpec(ratio=0.03, frequencies=[0.5, 8.0])
    assert d.type == "rayleigh"
    assert d.alpha == pytest.approx(2 * 0.03 * w1 * w2 / (w1 + w2))
    assert d.beta == pytest.approx(2 * 0.03 / (w1 + w2))
    d2 = DampingSpec(type="rayleigh", ratio=[0.02, 0.05], frequencies=[0.5, 8.0])
    zeta = lambda w: d2.alpha / (2 * w) + d2.beta * w / 2
    assert zeta(w1) == pytest.approx(0.02) and zeta(w2) == pytest.approx(0.05)
    with pytest.raises(Exception, match="go together"):
        DampingSpec(ratio=0.02)
    with pytest.raises(Exception, match="not both"):
        DampingSpec(ratio=0.02, frequencies=[1, 2], alpha=1.0)
    with pytest.raises(Exception, match="negative"):
        DampingSpec(ratio=[0.5, 0.001], frequencies=[1.0, 2.0])
    with pytest.raises(Exception, match="type: rayleigh"):
        DampingSpec(type="viscous", ratio=0.02, frequencies=[1, 2])
