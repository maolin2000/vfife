"""Gershgorin row sums: the explicit-stability bound, derived once.

WHAT THE NUMBER IS FOR
    Central difference with a lumped mass is stable while dt <= 2 / omega_max,
    and Gershgorin bounds omega_max**2 by

        max_i  ( sum_j |K_ij| ) / m_i

    so every element type has to hand the solver `sum_j |K_ij|` for each of
    its degrees of freedom. Under relaxation that same quantity IS the mass
    (Underwood): m_i = row_i / (4 s**2). Getting it too small does not warn
    anybody -- it under-masses a dof, and an under-massed dof in an explicit
    scheme is simply unstable, sometimes only after the run has already
    converged.

WHY IT LIVES HERE INSTEAD OF IN EACH ELEMENT
    It was written three times. `continuum.stiffness_row_sums` builds K and
    sums it, which is exact. `truss._emb_row_sums` derives the closed form for
    a bar projected onto host nodes, which is also exact -- its docstring
    records an earlier crude version that dropped a factor of up to 3.46 and
    took the reinforced beam to 1e140. `bonded.BondedTruss` derived it a third
    time and got HALF: it added the diagonal terms and omitted the equal
    off-diagonal ones, so the critical time step it declared was sqrt(2) too
    large, and the direction structure was dropped as well.

    Two of three independent derivations were wrong, in different ways, and
    each failure mode is silent. So it is derived here once and the elements
    call it.

THE TWO PIECES EVERY CASE NEEDS
    1.  DIRECTION. A stiffness that acts along a unit axis n and (optionally)
        across it is

            K = k_along (n (x) n) + k_across (I - n (x) n)

        whose row sum for dof i is, with |n|_1 = sum_j |n_j|,

            k_along |n_i| |n|_1  +  k_across sum_j |delta_ij - n_i n_j|

        The first term is what the third derivation dropped by using k_along
        isotropically: |n_i| |n|_1 reaches 1.207 for a bar at 37 degrees to an
        axis, so an isotropic k under-states the row by up to 21%, and by more
        once it is combined with the coupling factor below.

    2.  COUPLING. A stiffness written on a RELATIVE coordinate -- a spring
        between two points, a bar embedded by interpolation weights, an
        interface between a bar node and the host nodes under it -- assembles
        into more rows than it has. For a relative coordinate

            r = u_p - sum_a w_a u_a

        the row of dof p gains k (1 + sum_a |w_a|), and the row of host dof a
        gains k |w_a| (1 + sum_b |w_b|). Both carry the SAME factor
        (1 + sum |w|), which is 2 for the usual partition of unity. Dropping
        it is exactly the halving the third derivation suffered.

    Everything else is scatter.

THE THIRD PIECE, WHICH WAS LEFT OUT SEVEN TIMES
    The bound above is the UNDAMPED one. Every dynamic solver in the library
    set its step from it and then, on the next line, read the damping
    coefficients it had just ignored. Mass-proportional damping is harmless
    at the top of the spectrum -- zeta = alpha / (2 omega) vanishes as omega
    grows -- but Rayleigh's stiffness term is the opposite: zeta = beta omega
    / 2 is LARGEST at omega_max, and it is the top mode that decides the
    step. `damped_step_factor` below is the correction, derived once. Found
    on a lattice dome with beta = 0.001 that diverged from rest at the
    default step and was stable with beta = 0.
"""
from __future__ import annotations

import numpy as np


def damped_step_factor(omega_max, alpha=0.0, beta=0.0, coeff=0.0):
    """How much the explicit step must shrink because the damping is explicit.

    THE RECURRENCE. Leapfrog with the damping force taken from the LAGGED
    velocity -- which is how every solver here applies `alpha M v + beta K v`
    -- is, per mode, with a = omega dt and b = 2 zeta a,

        w' = (1 - b) w - a**2 x,      x' = x + w'

    whose amplification matrix has trace 2 - a**2 - b and determinant 1 - b.
    Both roots stay inside the unit circle exactly while

        a**2 + 2 b <= 4     i.e.     omega dt <= 2 (sqrt(1 + zeta**2) - zeta).

    So the undamped 2 / omega is multiplied by sqrt(1 + zeta**2) - zeta, which
    is 1 at zeta = 0 and ~1 / (2 zeta) once zeta is large. For Rayleigh beta
    that large-zeta limit is dt <= 2 / (beta omega_max**2): quadratic in the
    highest frequency, which is why a beta that is harmless on a coarse model
    stops a fine one dead.

    WHAT TO PASS. Only the terms the integrator applies EXPLICITLY. The
    solvers that treat the viscous coefficient by the averaged form

        v' = [(1 - c dt/2) v + dt a] / (1 + c dt/2)

    (truss2, the continuum) pass coeff = 0: for that form the determinant is
    (1 - b/2)/(1 + b/2) and the trace condition reduces to a**2 <= 4
    regardless of zeta -- the damping term itself is unconditionally stable
    and only the stiffness bound remains. Rayleigh is explicit everywhere.

    zeta is evaluated at omega_max, the Gershgorin one the caller already
    has. That is an over-estimate of the true top frequency, so the factor is
    conservative twice over -- once through omega itself and once through the
    beta omega / 2 inside zeta. A sharper omega_max would loosen it; a wrong
    one in the other direction is a run that diverges from rest.
    """
    omega = float(omega_max)
    if omega <= 0.0:
        return 1.0
    zeta = (float(alpha) + float(coeff)) / (2.0 * omega) + 0.5 * float(beta) * omega
    if zeta <= 0.0:
        return 1.0
    return float(np.sqrt(1.0 + zeta * zeta) - zeta)


def power_omega_max(K, mass, free, iterations: int = 300, tol: float = 1e-5,
                    seed: int = 0) -> float:
    """The top natural frequency of (K, M) on the free dofs by power
    iteration on M^-1/2 K M^-1/2 -- the number the Gershgorin row sums only
    bound from above.

    WHY IT IS WANTED. The undamped step is set from the Gershgorin bound and
    stays there: it is safe, cheap and the margin it carries is the margin a
    tangent that stiffens under load will eat. But the stiffness-proportional
    Rayleigh term shrinks the step by sqrt(1 + zeta^2) - zeta with
    zeta = beta omega / 2 evaluated AT the top frequency, and there the
    bound is paid twice -- once in omega, once in the zeta built on it.
    Gershgorin over-states omega_max of a frame by a factor 1.5 to 2.5, so
    with beta in play the step was 2 to 5 times smaller than the damped
    scheme needs. The true top frequency, converged from below, takes that
    back; the undamped part of the bound is not touched.

    Converged from below is the safe side here: the damped condition
    a^2 + 2 b <= 4 (a = omega dt, b = beta omega^2 dt) is met with a margin
    of three in a^2 by the undamped bound alone, so an omega under-estimated
    by even a factor two leaves b inside the limit. The iteration stops when
    the Rayleigh quotient moves by less than `tol`, or after `iterations`.
    """
    import scipy.sparse as sp
    free = np.asarray(free, bool)
    idx = np.flatnonzero(free)
    if idx.size == 0:
        return 0.0
    Kf = sp.csr_matrix(K)[idx][:, idx]
    d = 1.0 / np.sqrt(np.maximum(np.asarray(mass, float)[idx], 1e-300))
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(idx.size)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(int(iterations)):
        w = d * (Kf @ (d * v))
        new = float(v @ w)
        n = np.linalg.norm(w)
        if n <= 0.0:
            return 0.0
        v = w / n
        if lam > 0.0 and abs(new - lam) <= tol * abs(new):
            lam = new
            break
        lam = new
    return float(np.sqrt(max(lam, 0.0)))


def directional_row_sums(axis, k_along, k_across=0.0):
    """`sum_j |K_ij|` per dof for k_along (n x n) + k_across (I - n x n).

    `axis` is (n, 3) and need not be normalised; `k_along` and `k_across` are
    scalars or (n,). Returns (n, 3).

    Exact, not a bound: see the comment in the body for why summing the two
    parts separately (the triangle inequality) is not good enough.
    """
    n = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(n, axis=1)
    if np.any(norm < 1e-30):
        raise ValueError("a direction row sum needs non-degenerate axes")
    n = n / norm[:, None]
    a = np.abs(n)
    l1 = a.sum(axis=1, keepdims=True)                     # (n, 1)

    ka = np.broadcast_to(np.asarray(k_along, dtype=float).reshape(-1, 1),
                         (n.shape[0], 1))
    kt = np.asarray(k_across, dtype=float)
    if np.all(kt == 0.0):
        return ka * a * l1
    kt = np.broadcast_to(kt.reshape(-1, 1), (n.shape[0], 1))

    # EXACT, not the triangle inequality. Writing the combined matrix as
    #     K_ij = (k_a - k_t) n_i n_j + k_t delta_ij
    # separates cleanly: the off-diagonal entries carry only the first term,
    # and the diagonal carries both AND THEY CAN CANCEL. Summing |k_a n(x)n|
    # and |k_t (I - n(x)n)| separately is a valid bound but a loose one, and a
    # loose bound here is time step paid for nothing -- on a stiff transverse
    # penalty it came out 19% high.
    d = ka - kt
    off = np.abs(d) * a * (l1 - a)                        # j != i
    diag = np.abs(d * a * a + kt)                         # j == i
    return off + diag


def relative_coordinate_factor(*coefficient_blocks):
    """The L1 norm of the relative coordinate's row: `sum |c|` over EVERY dof.

    A stiffness written on `r = sum_p c_p u_p` assembles into every dof `r`
    touches, and the row of dof q picks up `k |c_q| * sum_p |c_p|`. This
    returns that `sum_p |c_p|`; the `|c_q|` is the caller's scatter.

    Pass one block per group of participating dofs, each (..., n_in_group);
    the sum runs over the last axis of each and adds across blocks. The two
    shapes in this package look different and are the same thing:

        interface   r = u_bar - sum_a w_a u_a
                    -> relative_coordinate_factor(ones, w) = 1 + sum|w|
        embedded    r = sum_b w1_b u_b - sum_a w0_a u_a
                    -> relative_coordinate_factor(w0, w1) = sum|w0| + sum|w1|

    Both are 2 for a partition of unity, which is why getting it wrong is easy
    and why returning 1 -- as `BondedTruss` did -- declares a critical time
    step sqrt(2) too large without looking obviously wrong.
    """
    total = None
    for block in coefficient_blocks:
        part = np.abs(np.asarray(block, dtype=float)).sum(axis=-1)
        total = part if total is None else total + part
    if total is None:
        raise ValueError("a relative coordinate needs at least one block")
    return total


def coupling_factor(weights):
    """`1 + sum_a |w_a|`: the common case of one own dof against weights."""
    w = np.asarray(weights, dtype=float)
    return relative_coordinate_factor(np.ones(w.shape[:-1] + (1,)), w)


def aligned_step(total_time: float, dt: float):
    """``(dt, n_steps)``. The callers keep the step they passed in as
    ``stable_dt``: the bound itself, which is what a probe of the stability
    limit reads (tests/test_rest_stays_at_rest.py).

    ``(dt, n_steps)`` with ``n_steps * dt == total_time`` and ``dt`` no
    larger than the stable step it is given.

    The dynamic loops label step ``i`` with ``t = i / n_steps * total_time``
    -- the instant a record or an amplitude is read at -- while the particles
    advance by ``dt``. With ``n_steps = round(total_time / dt)`` the two
    clocks disagree by up to half a step over the run: a relative time-scale
    error of ``dt / (2 total_time)``, seen as a frequency error. It showed on
    case_36, where the deck's phase against a closed form drifted by 2e-4
    of its frequency at a step of 2.8 ms over 6 s, and did not converge
    with the step (0.43, 1.6, 0.18, 0.24% for four halvings) because the
    rounding, not the step, set it."""
    if float(total_time) <= float(dt):
        # shorter than one stable step: one step of the stable size, as
        # before (the probes that read the bound back off `dt` build runs of
        # 1e-6 s for exactly this)
        return float(dt), 1
    n = int(np.ceil(float(total_time) / float(dt) * (1.0 - 1e-12)))
    return float(total_time) / n, n
